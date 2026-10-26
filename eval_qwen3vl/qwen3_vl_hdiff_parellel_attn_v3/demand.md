你现在是一个熟悉 PyTorch、HuggingFace Transformers、多模态大模型推理、视频 QA 评测流水线优化的工程助手。

我正在做 VideoMME / 视频 QA 推理评测。当前脚本名为 videomme_hdiff_parallel.py ，基础模型是 Qwen3-VL-8B 一类的视频多模态大模型。当前代码对每个样本串行执行：

1. process_vision_info(...)
2. processor(...)
3. inputs.to("cuda")
4. model.generate(...)
5. 保存结果

当前问题是：视频读取、解码、抽帧、resize、processor 打包主要在 CPU 和磁盘 IO 上执行；model.generate 在 GPU 上执行。现在外层是单线程 for 循环逐条样本处理，导致 GPU 利用率一会儿接近 0，一会儿拉满。目标是优化流水线，让 CPU 预处理和 GPU 推理尽可能重叠，提高整体推理吞吐。

你的任务是：基于现有videomme_hdiff_parallel.py，实现一个 DataLoader 版本的视频推理流水线。

核心目标：

1. 把每个样本的视频预处理逻辑封装进 PyTorch Dataset 的 __getitem__。
2. 使用 DataLoader 的 num_workers、prefetch_factor、pin_memory、persistent_workers 实现 CPU 多进程预处理和预取。
3. GPU 主进程只负责：
   - 从 DataLoader 获取已经预处理好的 inputs；
   - 把 tensor 移动到 cuda；
   - 调用 model.generate；
   - decode 输出；
   - 保存结果。
4. 尽量保持原始评测逻辑、prompt 构造逻辑、答案解析逻辑、结果保存格式不变。
5. 不要在 DataLoader worker 里加载大模型。模型只能在主进程/GPU 进程中加载一次。
6. DataLoader worker 里可以使用 processor，但如果 processor 在多进程中不可 pickle，需要改成在 worker 内部懒加载或通过 worker_init_fn 初始化。
7. 支持 batch_size=1 起步；如果代码结构允许，可以保留 batch_size > 1 的扩展接口，但不要强行复杂化。
8. 需要支持断点保存或逐条保存结果，避免长时间评测中途失败导致结果丢失。
9. 需要保证结果顺序可追踪，每个样本必须保留原始 index / question_id / video_id 等标识字段。
10. 推理时使用 torch.inference_mode()，并尽量使用 inputs.to("cuda", non_blocking=True)。

可使用的工具和技术：

- PyTorch Dataset
- PyTorch DataLoader
- num_workers
- prefetch_factor
- pin_memory
- persistent_workers
- torch.inference_mode()
- non_blocking=True
- tqdm
- json / jsonl 保存
- HuggingFace processor / tokenizer / model.generate
- 原始代码中的 process_vision_info
- 原始代码中的 prompt 构造函数、答案解析函数、保存函数

请你按照以下步骤完成：

第一步：阅读并理解现有 videomme_hdiff_parallel.py 的结构，找出以下逻辑：
- task_df 是如何构造的；
- 每个 row 如何构造 prompt / messages；
- process_vision_info 如何被调用；
- processor 如何处理 text/images/videos；
- inputs 如何传入 model.generate；
- 输出如何 decode；
- 结果如何保存。

第二步：重构代码，新增一个 Dataset，例如 VideoMMEDataset。它的职责是：
- 接收 task_df、processor、必要的路径参数和推理配置；
- 在 __getitem__(idx) 中读取一条样本；
- 构造 messages；
- 调用 process_vision_info；
- 调用 processor；
- 返回 inputs 和 metadata。

metadata 至少包含：
- idx
- video_id 或 video path
- question_id，如果原始数据中存在
- question
- answer / candidates，如果原始评测需要
- 其他保存结果所需字段

第三步：实现 collate_fn。
由于视频长度、文本长度可能不同，优先支持 batch_size=1。collate_fn 可以直接返回单个样本，避免复杂 padding。
如果支持 batch_size > 1，需要正确合并 processor 输出，并保证 padding 正确。

第四步：实现 DataLoader：
推荐初始参数为：

DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

要求把这些参数做成命令行参数或配置项：
- batch_size
- num_workers
- prefetch_factor
- pin_memory
- persistent_workers

第五步：改写主推理循环：
原始逻辑是：

for row in task_df.itertuples():
    preprocess
    inputs.to(cuda)
    model.generate
    save

请改成：

for batch in dataloader:
    inputs, metadata = batch
    inputs = move_to_cuda(inputs, device, non_blocking=True)
    with torch.inference_mode():
        generated_ids = model.generate(...)
    decode
    save result

要求实现一个通用函数 move_to_cuda，可以递归处理 dict / list / tuple / tensor：

def move_to_cuda(obj, device="cuda", non_blocking=True):
    ...

第六步：增加基础性能统计。
至少统计：
- 总样本数
- 总耗时
- 平均每样本耗时
- samples/sec
- 可选：每 N 条打印一次速度

不要只看 GPU utilization，要以 samples/sec 作为主要指标。

第七步：保证可复现实验。
如果原始代码使用采样生成，请保留参数；但默认建议使用：
- do_sample=False
- temperature=None
- top_p=None

如果必须保持原设置，请不要擅自改变评测结果。

第八步：处理异常。
如果某个样本视频读取失败、processor 失败或 generate 失败：
- 不要让整个评测直接崩掉；
- 记录该样本的 idx、video_id、错误信息；
- 保存到 error log；
- 继续处理下一个样本。

但如果是 CUDA OOM，需要：
- 打印当前样本 metadata；
- 清理 torch.cuda.empty_cache()；
- 记录错误；
- 继续或安全退出，由代码参数控制。

交付条件：

1. 给出修改后的完整代码，或者给出清晰的 patch/diff。
2. 保留原始 videomme_hdiff_parallel.py 的主要评测行为，不要改变数据读取、prompt 格式、答案解析和结果保存逻辑，除非必须。
3. 明确标注新增的类和函数：
   - VideoMMEDataset
   - collate_fn
   - move_to_cuda
   - build_dataloader
   - run_inference_with_dataloader
4. 代码需要可以直接运行。
5. 给出推荐启动命令，例如：

python videomme_hdiff_parallel.py \
    --num_workers 4 \
    --prefetch_factor 2 \
    --batch_size 1 \
    --pin_memory \
    --persistent_workers

6. 给出调参建议：
   - num_workers 从 2、4、8 试；
   - prefetch_factor 从 2 开始；
   - batch_size 先用 1；
   - 如果 CPU 内存占用过高，降低 num_workers 或 prefetch_factor；
   - 如果 GPU 仍然等待，增加 num_workers；
   - 如果磁盘 IO 打满，增加 worker 没有意义。

7. 给出验证方式：
   - 对比重构前后前 20 条样本的输出是否一致；
   - 对比总耗时和 samples/sec；
   - 确认结果文件格式不变；
   - 确认没有重复样本或漏样本；
   - 确认出错样本被记录到 error log。

注意事项：

- 不要在多个 worker 中加载 GPU model。
- 不要把大模型传进 Dataset 或 DataLoader worker。
- 如果 processor 不能被 pickle，需要在 Dataset 内部延迟初始化，或者在每个 worker 内部单独初始化 processor。
- 如果使用 CUDA，必须避免在 DataLoader worker 里调用 cuda。
- DataLoader worker 只做 CPU 预处理。
- 主进程负责所有 GPU 推理。
- 如果 batch_size=1，collate_fn 尽量简单，稳定优先。
- 代码优先保证正确性和稳定性，再考虑复杂优化。


可以适度重构代码结构，把数据读取、预处理、推理、保存、性能统计拆成独立函数。要求代码清晰、可维护，便于后续加入视频帧缓存、动态 FPS、帧筛选和 batch 推理。