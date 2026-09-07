# 摄像头并发重构（作业三）

把单线程摄像头程序重构为「后台采集线程 + 主线程界面」的并发组件：设备、会话
与预览界面彼此解耦，并且核心逻辑可以用不依赖真实摄像头的假帧源做自动测试。

## 构建与运行

从仓库根目录构建（摄像头作业默认关闭，需显式打开）：

```bash
cmake -S . -B build-camera \
  -DSTURDY_GUIDE_BUILD_CAMERA_HOMEWORK=ON -DBUILD_TESTING=ON
cmake --build build-camera
ctest --test-dir build-camera --output-on-failure
```

运行（参数与起始版本一致）：

```bash
./build-camera/sturdy-guide-camera --device 0 --width 1280 --height 720
```

- 按 `S` 把当前画面（含叠加信息）保存到 `captures/`；
- 按 `Q` 或关闭窗口退出。

## 目录结构

```text
camera_monitor/
├── starter/main.cpp       原始单线程版本（保留对照，不再参与构建）
├── include/camera/        公共接口：FrameSource 及各类实现、CameraSession
├── src/                   核心库 sturdy_guide_camera_core 的实现
├── app/                   可执行程序 sturdy-guide-camera（界面与组装）
├── tests/                 CTest 注册的硬件无关测试
└── README.md              本文件
```

## 重构前后职责对比

| 职责 | 起始版本 | 重构后 |
| --- | --- | --- |
| 参数解析与校验 | `main()` 内联 | `app/main.cpp::parse_options` |
| 打开真实摄像头 | `main()` 内联 | `OpenCvCamera`（实现 `FrameSource`） |
| 采集循环与线程生命周期 | 无（串行读取） | `CameraSession` 后台线程 |
| 帧缓冲与丢帧策略 | 无 | `CameraSession` 有界两帧缓冲 |
| 帧率统计、叠加文字 | `main()` 内联 | `PreviewApplication` |
| 窗口、键盘、截图 | `main()` 内联 | `PreviewApplication` |
| 错误处理 | `main()` 的 try/catch | 打开失败立即抛出；运行期错误由主线程观察并报告 |
| 测试性 | 依赖真实摄像头 | `FakeFrameSource` + CTest |

## 线程安全契约

### 1. 资源所有权与生命周期

- `OpenCvCamera` 独占一个 `cv::VideoCapture`；它由 `CameraSession` 通过
  `std::unique_ptr<FrameSource>` 唯一持有，生命周期与会话相同。
- `CameraSession` 独占 `std::thread worker_`、`mutex_`、`condition_`、
  有界缓冲和 `exception_`，并在析构时执行 `stop()` + `join()`。
- `PreviewApplication` 只使用调用线程（主线程），不拥有设备；窗口由 OpenCV
  highgui 管理，只允许主线程调用 `imshow` / `waitKey`。
- `main()` 以栈对象顺序保证：先构造设备，再构造会话，最后运行预览；退出时
  先 `stop()` 再 `join()`。

### 2. 状态机

`CameraSession` 的状态：

```text
Idle ──start()──> Running ──stop()──> Stopping ──线程结束──> Stopped
  ▲                                                          │
  └──────────────────── join() 后可以再次 start() ────────────┘
```

- `Idle`：尚未启动，或工作线程已结束并被 `join()`；
- `Running`：`capture_loop` 正在采集；
- `Stopping`：已请求停止（`stopping_ = true`），工作线程退出前；
- `Stopped`：工作线程已退出（`running_ = false`），缓冲可能仍有未消费帧。

合法转换：只有 `Idle` 可以 `start()`；`stop()` 幂等，可从任何状态调用；重复
`start()`（线程仍 joinable 或仍在运行）抛出 `std::logic_error`。

### 3. 哪些方法可以跨线程调用

- `latest_frame()`、`wait_for_frame()`、`exception()`、`buffered_frames()`、
  `is_running()` 内部加锁，允许任意线程并发调用；
- `start()` / `join()` 属于生命周期操作，按状态机由单个所有者线程（主线程）
  调用；
- `stop()` 也可安全地从其他线程调用（先置位再 `notify_all`），主线程负责与
  `join()` 配对。

### 4. mutex 保护什么

`mutex_` 保护 `buffer_`、`exception_`、`stopping_`、`running_` 以及它们共同
维护的不变量：

- `buffer_.size() <= 2`，缓冲中不存在空帧；
- 缓冲中帧按采集顺序排列，最新帧在尾部；
- `running_ == true` 期间恰好有一个工作线程在执行 `capture_loop`；
- `exception_` 只在工作线程结束时写入，可被任意线程读取。

`worker_` 不放在锁内：`start()` / `join()` 由单一所有者按状态机串行调用，
加锁反而会让「先置 stopping 再 join」与 `capture_loop` 产生不必要的耦合。

### 5. 缓冲满时丢弃哪一帧

丢弃最旧的队首帧，把新帧放到队尾。显示线程只关心“最新画面”，旧帧越积越多的
唯一后果是内存和延迟增长；保留最新帧让界面永远能看到最近一次采集结果。

### 6. 为什么不能在持锁时做慢速 I/O

`VideoCapture::read()` 可能阻塞（设备繁忙、断流重连），`imshow()` 需要配合
窗口事件循环，`imwrite()` 是磁盘写入。若在临界区内执行，采集线程或主线程会
被同一把锁卡死，有界缓冲失去意义。因此采集读取发生在工作线程的锁外，界面与
截图发生在主线程的锁外（`latest_frame()` 返回克隆帧，不持有锁进行显示）。

### 7. 后台异常如何越过线程边界

`capture_loop` 用 try/catch 捕获任何异常，把 `std::current_exception()` 存入
`exception_`（加锁写），随后把 `running_` 置为 `false` 并唤醒等待者。主线程
通过 `wait_for_frame()` 的唤醒谓词或轮询 `exception()` 观察到异常，再用
`std::rethrow_exception` 抛回主线程，由 `main()` 统一输出错误。

### 8. 析构时 stop / 唤醒 / join 的顺序

```text
stop()：加锁置 stopping_ = true，解锁后 notify_all()
join()：阻塞直到 capture_loop 退出
```

先发停止信号再等待线程结束，可以保证：

- 正在读帧的线程在本次 `read()` 返回后看到 `stopping_` 并退出；
- 在条件变量上等待的消费者被 `notify_all()` 唤醒；
- 析构返回时工作线程必然已经结束，不存在悬空线程访问已析构的会话。

## 本地自动测试

`camera_session_test` 完全基于 `FakeFrameSource`，不会打开真实设备 `0`，覆盖：

1. `start()` 后能取得帧，帧值随读取次数递增；
2. 重复 `start()` 抛出 `std::logic_error`（契约见上文状态机）；
3. 连续调用 `stop()` 不崩溃、不死锁；
4. 析构/`join()` 保证工作线程先结束（测试对象在作用域结束时析构）；
5. 消费较慢时缓冲始终不超过两帧；
6. `FakeFrameSource` 抛出的读取错误能在主线程通过 `exception()` 观察到；
7. 空帧不会被发布为有效画面。

## 真机集成测试记录

| 项目 | 结果 |
| --- | --- |
| 操作系统 | Windows 10（版本 10.0.19045.6466，家庭中文版） |
| 摄像头 | hm1091_techfront（`USB\VID_0408&PID_1020`，`Status=OK` / `CM_PROB_NONE`） |
| 分辨率 | 1280x720（`captures/capture-1.png` 实测） |
| 构建 | `C:\rebuild.bat` 增量构建，EXIT=0 |
| CTest | 4/4 通过 |

> 自动测试不代表摄像头可用；真机验证结论见上表，并在 PR 中附带运行截图。
