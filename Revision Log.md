# Revision Log

## implicit_q_learning

### extract_feature.py

| 修改行数 | 修改前                                          | 修改后                                                                          |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------- |
| 31       | 单个 .pkl 文件路径                              | the path to a single .pkl file                                                  |
| 39       | 加载图像编码器                                  | load the image encoder                                                          |
| 63       | 读取单个 pkl 文件，包含多个 demo                | read a single .pkl file of multiple demonstrations                              |
| 75       | 处理每一条轨迹                                  | process each trajectory                                                         |
| 93       | 检查长度一致性                                  | check the length consistency                                                    |
| 99       | 定义图像转 tensor 函数（支持 HWC/CHW 自动识别） | define a function to convert image to tensor (support HWC/CHW auto-recognition) |
| 124      | 分批编码                                        | encode in batches                                                               |
| 142      | 构建                                            | build                                                                           |
| 144      | 当前状态                                        | current state                                                                   |
| 151      | 原始图像（假设为 CHW -> 转为 HWC 存储更通用）   | original image (assuming CHW -> convert to HWC for better storage)              |
| 162      | 构造 observation 字典                           | construct the observation dictionary                                            |
| 182      | 构建最终 dataset                                | build the final dataset                                                         |
| 180      | 使用原始 dones，或 fallback 到是否最后一步      | use the original dones, with a fallback to checking if it's the final step      |
| 188      | 标记 episode 是否结束                           | check and tag episode end status                                                |
| 191      | 保存结果                                        | save the results                                                                |

## scrip

### assembly_gym.py

| 修改行数 | 修改前                 | 修改后                                                 |
| -------- | ---------------------- | ------------------------------------------------------ |
| 9        | 用于图像处理           | image-processing                                       |
| 263      | 设置位置               | set position                                           |
| 425      | 获取摄像机观察数据     | acquire camera observation data                        |
| 428      | 获取RGB图像            | acquire an RGB image                                   |
| 429      | 获取深度图像           | acquire a depth image                                  |
| 432      | 只取RGB，忽略Alpha通道 | extract the RGB channels, discarding the alpha channel |
| 435      | 获取摄像机数据时出错   | An error occurred while acquiring camera data image    |

---

### assembly_task.py

| 修改行数 | 修改前                         | 修改后                                                                                |
| -------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| 15       | 自定义任务类，继承自 BaseTask  | define a custom task class that inherits from the BaseTask class                      |
| 88       | 添加支撑件                     | add support components                                                                |
| 89       | 支撑件高度，可在obj_info中配置 | the height of the support component, which is configurable within the obj_info object |
| 105      | 支撑件尺寸                     | dimensions of the support component                                                   |
| 108      | 支撑件缩放比例                 | scaling factor of the support component                                               |
| 112      | 添加物体，放在支撑件上         | cadd an object, positioning it on the support component                               |
| 221      | 支撑件高度，可在obj_info中配置 | the height of the support component, which is configurable within the obj_info object |

---

### disassembly_gym.py

| 修改行数 | 修改前                 | 修改后                                                 |
| -------- | ---------------------- | ------------------------------------------------------ |
| 259      | 设置位置               | set position                                           |
| 403      | 获取RGB图像            | acquire an RGB image                                   |
| 404      | 获取深度图像           | acquire a depth image                                  |
| 407      | 只取RGB，忽略Alpha通道 | extract the RGB channels, discarding the alpha channel |
| 410      | 获取摄像机数据时出错   | An error occurred while acquiring camera data image    |

---

### disassembly_task.py

| 修改行数 | 修改前                        | 修改后                                                           |
| -------- | ----------------------------- | ---------------------------------------------------------------- |
| 17       | 自定义任务类，继承自 BaseTask | define a custom task class that inherits from the BaseTask class |

---

### replay_logging_assemble_failed.py

| 修改行数 | 修改前                                       | 修改后                                                                                      |
| -------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 48       | 提取时间戳并排序                             | extract and sort timestamps                                                                 |
| 51       | 按照 '_' 分割，第一个部分就是时间戳          | split the string by the underscore character ('_'), and the first segment is the timestamp  |
| 55       | 如果无法解析时间戳，则返回一个极大值排到最后 | if the timestamp cannot be parsed, assign a maximum value to ensure it is sorted to the end |
| 88       | 获取所有文件名（仅文件名，不是完整路径）     | retrieve all file names (only the names, not the full paths)                                |
| 93       | 排序后的文件名列表                           | the sorted list of file names                                                               |
| 95       | 构建完整路径                                 | build the complete file path                                                                |
| 98       | 创建 HDF5 文件                               | create an HDF5 file                                                                         |
| 100      | 创建 group 存储所有数据帧                    | create a group for storing all data frames                                                  |

### replay_logging_assemble.py

| 修改行数 | 修改前                                       | 修改后                                                                                      |
| -------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 46       | 提取时间戳并排序                             | extract and sort timestamps                                                                 |
| 49       | 按照 '_' 分割，第一个部分就是时间戳          | split the string by the underscore character ('_'), and the first segment is the timestamp  |
| 53       | 如果无法解析时间戳，则返回一个极大值排到最后 | if the timestamp cannot be parsed, assign a maximum value to ensure it is sorted to the end |
| 86       | 获取所有文件名（仅文件名，不是完整路径）     | retrieve all file names (only the names, not the full paths)                                |
| 91       | 排序后的文件名列表                           | the sorted list of file names                                                               |
| 93       | 构建完整路径                                 | build the complete file path                                                                |
| 96       | 创建 HDF5 文件                               | create an HDF5 file                                                                         |
| 98       | 创建 group 存储所有数据帧                    | create a group for storing all data frames                                                  |

---

### replay_logging_disassemble_failed.py

| 修改行数 | 修改前                                       | 修改后                                                                                      |
| -------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 47       | 提取时间戳并排序                             | extract and sort timestamps                                                                 |
| 50       | 按照 '_' 分割，第一个部分就是时间戳          | split the string by the underscore character ('_'), and the first segment is the timestamp  |
| 54       | 如果无法解析时间戳，则返回一个极大值排到最后 | if the timestamp cannot be parsed, assign a maximum value to ensure it is sorted to the end |
| 85       | 获取所有文件名（仅文件名，不是完整路径）     | retrieve all file names (only the names, not the full paths)                                |
| 90       | 排序后的文件名列表                           | the sorted list of file names                                                               |
| 92       | 构建完整路径                                 | build the complete file path                                                                |
| 95       | 创建 HDF5 文件                               | create an HDF5 file                                                                         |
| 97       | 创建 group 存储所有数据帧                    | create a group for storing all data frames                                                  |

### replay_logging_disassemble.py

| 修改行数 | 修改前                                       | 修改后                                                                                      |
| -------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 47       | 提取时间戳并排序                             | extract and sort timestamps                                                                 |
| 50       | 按照 '_' 分割，第一个部分就是时间戳          | split the string by the underscore character ('_'), and the first segment is the timestamp  |
| 54       | 如果无法解析时间戳，则返回一个极大值排到最后 | if the timestamp cannot be parsed, assign a maximum value to ensure it is sorted to the end |
| 85       | 获取所有文件名（仅文件名，不是完整路径）     | retrieve all file names (only the names, not the full paths)                                |
| 90       | 排序后的文件名列表                           | the sorted list of file names                                                               |
| 92       | 构建完整路径                                 | build the complete file path                                                                |
| 95       | 创建 HDF5 文件                               | create an HDF5 file                                                                         |
| 97       | 创建 group 存储所有数据帧                    | create a group for storing all data frames                                                  |

---

## utils

### convert_h5_pkl.py

| 修改行数 | 修改前                                  | 修改后                                                                                                                                                                                                                                                                                                                               |
| -------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 24-31    |                                         | Convert custom HDF5 expert data to standard .pkl format (for IL training)Args:h5_path (str): Path to the input .h5 fileoutput_dir (str): Output directory. If None, use the same directory as the h5 filesave_per_demo (bool): Whether to save each demonstration as a separate .pkl file; otherwise save as a single demos.pkl file |
| 43       | H5 内没有 '/data_frames' 分组           | The HDF5 file lacks the '/data_frames' group                                                                                                                                                                                                                                                                                         |
| 49       | 按名称排序                              | sort by name                                                                                                                                                                                                                                                                                                                         |
| 57       | 提取基本字段                            | extract basic fields                                                                                                                                                                                                                                                                                                                 |
| 62       | 构建 dones: 最后一步为 True，其余 False | build the dones array where only the final step is True                                                                                                                                                                                                                                                                              |
| 64       | 最后一步                                | the final step                                                                                                                                                                                                                                                                                                                       |
| 66       | 构建 obs 字典（可根据需要增减字段）     | construct the obs dictionary (fields can be added or removed as needed)                                                                                                                                                                                                                                                              |
| 68       | 连续状态向量                            | continuous state vector                                                                                                                                                                                                                                                                                                              |
| 73       | 离散状态向量                            | discrete state vector                                                                                                                                                                                                                                                                                                                |
| 77       | 图像观测（可选，注意内存）              | image observation (optional, note memory usage)                                                                                                                                                                                                                                                                                      |
| 82       | 将所有状态观测拼接成一个数组            | concatenate all state observations into one array                                                                                                                                                                                                                                                                                    |
| 109      | 字典形式的观测                          | observations in the form of a dictionary                                                                                                                                                                                                                                                                                             |
| 113      | 可扩展其他字段                          | extensible to other fields                                                                                                                                                                                                                                                                                                           |
| 118      | 如果选择每个 demo 单独保存              | If the option to save each demonstration individually is selected                                                                                                                                                                                                                                                                    |
| 125      | 保存所有 demos 到一个文件               | save all demonstrations into a single file                                                                                                                                                                                                                                                                                           |

---

### device_control.py

| 修改行数         | 修改前                                                                                           | 修改后                                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 16               | 控制频率 (Hz) 太大会导致按一次，动多次现象，太低会导致延迟                                       | an excessively high control frequency (Hz) will cause multiple movements per single command, while an excessively low frequency will introduce significant latency        |
| 18               | 初始化速度参数                                                                                   | initialize the velocity parameters                                                                                                                                        |
| 19               | 平移速度                                                                                         | translational velocity                                                                                                                                                    |
| 20               | 旋转速度                                                                                         | rotational velocity                                                                                                                                                       |
| 22               | 初始化 Pygame 键盘系统                                                                           | initialize the Pygame keyboard system                                                                                                                                     |
| 24               | 必须初始化显示才能使用键盘                                                                       | must initialize the display to use the keyboard                                                                                                                           |
| 26               | 键盘控制说明                                                                                     | Keyboard Control Instructions                                                                                                                                             |
| 27               | 平移控制: W/S(前后), A/D(左右), 上下方向键(上下)                                                 |                                                                                                                                                                           |
| 28               | 旋转控制: J/L(绕X轴), I/K(绕Y轴), U/O(绕Z轴)                                                     | Rotation Controls: J/L (Rotation about X-axis), I/K (Rotation about Y-axis), U/O (Rotation about Z-axis)                                                                  |
| 29               | 夹爪控制: 空格键(关闭), Ctrl键(打开)                                                             | Gripper Control: Spacebar (for closing), Ctrl key (for opening)                                                                                                           |
| 30               | 功能键: X(重置世界并清除数据), Y(停止采集并保存), ESC(退出不保存)                                | Function Keys: X (Reset the world and clear data); Y (Stop collection and save); ESC (Exit without saving)                                                                |
| 31               | 数据采集: 按任意动作键停止采集数据                                                               | Data Collection: Press any movement key to stop data acquisition                                                                                                          |
| 32               | 功能键：F(调节速度,xyz方向加快5倍,角度方向加快5倍)，R(调节速度,xyz方向减慢50倍,角度方向减慢10倍) | Function Keys: F (Adjust speed: increases XYZ direction by 5x and angular direction by 5x); R (Adjust speed: decreases XYZ direction by 50x and angular direction by 10x) |
| 33               | 现在的速度为                                                                                     | Current speed                                                                                                                                                             |
| 35               | 启动线程                                                                                         | start the thread                                                                                                                                                          |
| 47               | 边缘检测：只在按下瞬间触发                                                                       | edge detection: Triggered only on the rising edge (the moment the key is pressed)                                                                                         |
| 55,63            | 速度调节为                                                                                       | Speed adjustment is set to                                                                                                                                                |
| 67               | 构造 action                                                                                      | construct the action                                                                                                                                                      |
| 72               | 平移控制                                                                                         | translational control                                                                                                                                                     |
| 80               | 旋转控制                                                                                         | rotational control                                                                                                                                                        |
| 88               | 夹爪控制                                                                                         | gripper control                                                                                                                                                           |
| 97               | 功能键                                                                                           | function keys                                                                                                                                                             |
| 102              | 构建完整 action                                                                                  | build the complete action                                                                                                                                                 |
| 105              | 放入队列                                                                                         | put the action into the queue                                                                                                                                             |
| 108              | 防止堆积                                                                                         | prevent accumulation                                                                                                                                                      |
| 111              | 控制频率                                                                                         | control frequency                                                                                                                                                         |
| 120              | 默认配置参数（可以被覆盖）                                                                       |                                                                                                                                                                           |
| 122              | 平移速度                                                                                         | translational velocity                                                                                                                                                    |
| 123              | 旋转速度                                                                                         | rotational velocity                                                                                                                                                       |
| 124              | 控制频率 (Hz) 太大会导致按一次，动多次现象，太低会导致延迟                                       | an excessively high control frequency (Hz) will cause multiple movements per single command, while an excessively low frequency will introduce significant latency        |
| 130              | 控制说明                                                                                         | Control Instructions                                                                                                                                                      |
| 131              | 左摇杆：前后/左右移动                                                                            | Left Joystick: For forward/backward and left/right movement                                                                                                               |
| 132              | 右摇杆上下：上下移动，右摇杆左右：绕X轴旋转                                                      | Right Joystick: Up/Down for vertical movement; Left/Right for rotation about the X-axis                                                                                   |
| 133              | D-Pad：左右绕Y轴旋转，上下绕Z轴旋转                                                              | D-Pad: Left/Right for rotation about the Y-axis; Up/Down for rotation about the Z-axis                                                                                    |
| 134              | A键：张开夹爪，B键：闭合夹爪                                                                     | A button: Open gripper; B button: Close gripper                                                                                                                           |
| 135              | Y键：退出此次收集, X键：重置世界并清除数据                                                       | Y button: Exit the current collection; X button: Reset the world and clear data                                                                                           |
| 136              | 左手的激发键（4键）：退出整个程序                                                                | Left-hand trigger button (Button 4): Exit the entire program                                                                                                              |
| 137              | 按键6：调节速度，加快5倍，按键7：调节速度，减慢5倍                                               | Button 6: Adjust speed - increase by 5x; Button 7: Adjust speed - decrease by 5x                                                                                          |
| 141              | 启动线程                                                                                         | start the thread                                                                                                                                                          |
| 161,169          | 速度调节为                                                                                       | speed adjustment is set to                                                                                                                                                |
| 171              | 读取摇杆输入                                                                                     | read the joystick input                                                                                                                                                   |
| 172              | 左摇杆 X                                                                                         | Left Joystick X-Axis                                                                                                                                                      |
| 173              | 左摇杆 Y                                                                                         | Left Joystick Y-Axis                                                                                                                                                      |
| 174              | 右摇杆 X                                                                                         | Right Joystick X-Axis                                                                                                                                                     |
| 175              | 右摇杆 Y                                                                                         | Right Joystick Y-Axis                                                                                                                                                     |
| 178              | 左右（绕Y轴）                                                                                    | Left/Right (about Y-axis)                                                                                                                                                 |
| 179              | 上下（绕Z轴）                                                                                    | Up/Down (about Z-axis)                                                                                                                                                    |
| 181，182,183,184 | 键                                                                                               | button                                                                                                                                                                    |
| 185              | 不保存按钮                                                                                       | the "do not save" button                                                                                                                                                  |
| 187              | 更新共享变量                                                                                     | update the shared variables                                                                                                                                               |
| 193              | 夹爪状态                                                                                         | gripper state                                                                                                                                                             |
| 199              | 构造动作向量                                                                                     | formulate the action vector                                                                                                                                               |
| 204              | 前后                                                                                             | forward/backward                                                                                                                                                          |
| 205              | 左右                                                                                             | left/right                                                                                                                                                                |
| 206              | 上下                                                                                             | up/down                                                                                                                                                                   |
| 207              | 绕X轴                                                                                            | about the X-axis                                                                                                                                                          |
| 208              | 绕Y轴                                                                                            | about the Y-axis                                                                                                                                                          |
| 209              | 绕Z轴                                                                                            | about the Z-axis                                                                                                                                                          |
| 213              | 放入队列                                                                                         | put the action into the queue                                                                                                                                             |
| 215              | 防止队列堆积                                                                                     | prevent queue accumulation                                                                                                                                                |
| 218              | 控制频率                                                                                         | control frequency                                                                                                                                                         |
| 220              | 工具函数：应用死区                                                                               | Utility function: Apply dead zone                                                                                                                                         |
| 222              | 应用死区，消除手柄漂移影响                                                                       | apply a dead zone to eliminate joystick drift                                                                                                                             |
| 233              | 未检测到游戏手柄                                                                                 | No game controller detected                                                                                                                                               |
| 237              | 已连接手柄                                                                                       | Controller connected                                                                                                                                                      |

---

### inspect_h5file.py

| 修改行数 | 修改前               | 修改后                                         |
| -------- | -------------------- | ---------------------------------------------- |
| 3        | 递归打印 H5 文件结构 | recursively print the H5 file structure        |
| 9        | 检查文件是否存在     | check for the existence of the file            |
| 11       | 错误：文件  不存在   | Error: File  not exist                         |
| 16       | 正在解析 H5 文件     | Currently parsing the H5 file                  |
| 19       | 递归遍历文件结构     | traverse the entire file structure recursively |
| 24       | 数据集   形状  类型  | data set     shape        type                 |
| 26       | 组                   | group                                          |
| 27       | 打印属性（如果有）   | print the attributes if they exist             |
| 30       | 属性                 | attribute                                      |
| 31       | 递归进入子组         | recursively descend into subgroups             |
| 35       | 开始打印结构         | start printing the structure                   |
| 39       | 读取文件时发生错误   | an error occurred while reading the file       |
| 41       | 主程序               | main program                                   |

### inspect_pkl.py

| 修改行数 | 修改前                              | 修改后                                                     |
| -------- | ----------------------------------- | ---------------------------------------------------------- |
| 11       | 文件不存在                          | file not exist                                             |
| 18       | 无法加载 PKL 文件                   | unable to load PKL file                                    |
| 21       | 打印整体类型                        | print the overall type                                     |
| 22       | 文件路径                            | file path                                                  |
| 23       | 数据类型                            | data type                                                  |
| 25       | 如果是 list（比如多个 demo）        | If it is a list (e.g., containing multiple demonstrations) |
| 27       | 数据是列表，包含 {len(data)} 个元素 | The data is a list, containing {len(data)} elements        |
| 28       | 只打印前3个 demo 示例               | print only the first 3 demo examples                       |
| 29       | 类型                                | type                                                       |
| 33       | 内容                                | content                                                    |
| 35       | 还有 {len(data) - 3} 个未显示...    | ... and {len(data) - 3} more items are not displayed       |
| 37       | 如果是字典（单个 demo）             | if it is a dictionary (single demo)                        |
| 41       | 其他类型                            | other types                                                |
| 43       | 内容                                | content                                                    |
| 47       | 键值对结构                          | key-value structure                                        |
| 61       | 修改成你的文件路径                  | modify to your file path                                   |
| 64       | 如果是目录，遍历所有 .pkl 文件      | if it is a directory, traverse all .pkl files              |

---

### utils.py

| 修改行数 | 修改前                                              | 修改后                                                                        |
| -------- | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| 65       | 力的阈值（绝对值）                                  | force threshold (absolute value)                                              |
| 74       | 启动 GUI 线程                                       | start the GUI thread                                                          |
| 66       | 同一对象的提醒最小间隔（秒）                        | minimum reminder interval (in seconds) for the same object                    |
| 81       | 检查 observation 中所有以 'contact_forces' 结尾的键 | check all keys in the observation that end with 'contact_forces'              |
| 91       | 判断是否超过阈值                                    | check if the force exceeds the threshold                                      |
| 99       | 打印警告 + 投递 GUI 弹窗                            | print a warning and dispatch a GUI popup                                      |
| 109      | GUI 专用线程：Tk 主循环 + 队列轮询                  | GUI-specific thread: Tk main loop + queue polling                             |
| 129      | GUI 线程执行：显示新弹窗                            | GUI thread execution: display new popup                                       |
| 130      | 关闭旧弹窗                                          | close the old popup                                                           |
| 144      | 力显示：取前三个分量，若不足补零                    | force display: take the first three components, if not enough, pad with zeros |
| 156      | 1.5s 后自动关闭                                     | 1.5 seconds later, the popup will automatically close                         |

## run_main.py

| 修改行数 | 修改前                           | 修改后                                                               |
| -------- | -------------------------------- | -------------------------------------------------------------------- |
| 122      | 启动手柄线程                     | start the joystick thread                                            |
| 123      | 使用列表实现可变布尔标志         | use a list to implement a mutable boolean flag                       |
| 125      | 共享变量                         | shared variable                                                      |
| 129      | 用于标记是否需要保存             | used to flag whether saving is required                              |
| 130      | 夹爪状态                         | gripper status                                                       |
| 133      | 创建锁                           | create a lock                                                        |
| 176      | 应用动作                         | apply the action                                                     |
| 180      | 等待20步，确保保存完整的动作序列 | wait for 20 steps to ensure that the entire action sequence is saved |
