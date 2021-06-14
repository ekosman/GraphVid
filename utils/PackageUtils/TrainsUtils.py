from trains import Task


def get_trains_logger(project_name, task_name):
    task = Task.init(project_name=project_name, task_name=task_name)
    logger = task.get_logger()
    return logger
