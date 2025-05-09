from evalscope import TaskConfig
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        'data': ['MMBench-Video'],
        'limit': 10,
        'mode': 'all',
        'model': [ 
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.0,
            'type': 'Qwen2.5-VL-7B-Instruct',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 1024,}
            ],
        'reuse': False,
        'nproc': 4,
        'judge': 'exact_matching'}
)


def run_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()
