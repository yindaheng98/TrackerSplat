# TrackerSplat 重构计划

TrackerSplat抽象为一个通用的框架, 专用于实现根据一些外部信息移动Gaussian点的算法, 本项目提供统一的数据读写渲染测试后处理等周边功能

Point Tracker为其中的一种实现, 用训练移动Gaussian点是另一种实现, 还可以有其他的实现方式

Point Tracker实现单独拎出来作为一个库实现, 在TrackerSplat里面调用

refiner也是Point Tracker里的

基于训练的TrackerSplat实现还是留在本项目里

trackersplat/incrementaltraining.py移动到基于基于训练的TrackerSplat实现中作为测试代码

trackersplat/pointtracking.py移动到基于基于Point Tracker的TrackerSplat实现中作为测试代码


fixed view的抽象是给Point Tracker用的, 因为Point Tracker推断需要视角静止, 应该放到Point Tracker库里

基于训练的TrackerSplat实现应该是无所谓fixed view和dynamic view的, 应该做成通用的, 直接基于MotionEstimator实现, 通过参数调节控制几帧更新一次

TrackerSplat里应该包含batch func的实现, 因为几个baseline的实现需要用到后帧加点影响前帧, 但是不包含fixed view的实现, 这就需要修改fixed view和batch func的继承顺序, 应该是先有batch func再有fixed view
