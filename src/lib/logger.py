import os
import traceback
from datetime import datetime

LOGGER = None

def log_function(func):
    def try_call_log(*args, **kwargs):
        try:
            if(LOGGER is not None):
                message = f"Calling: {func.__name__}..."
                LOGGER.log_info(message=message, message_type="info")
            return func(*args, **kwargs)
        except Exception as e:
            if(LOGGER is None):
                raise e
            message = traceback.format_exc()
            print_(message, message_type="error")
            raise
    return try_call_log

def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def print_(message, message_type="info"):
    print(message)
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return

def log_info(message, message_type="info"):
    if(LOGGER is not None):
        LOGGER.log_info(message, message_type)
    return

class Logger():
    def __init__(self, exp_path, file_name="logs.txt"):
        logs_path = os.path.join(exp_path, file_name)
        self.logs_path = logs_path

        if not os.path.exists(logs_path):
            if(not os.path.exists(exp_path)):
                os.makedirs(exp_path)
            with open(logs_path, 'w') as f:
                f.write("")

        global LOGGER
        LOGGER = self
        return

    def log_info(self, message, message_type="info", **kwargs):
        if(message_type not in ["new_exp", "info", "warning", "error", "params"]):
            message_type = "info"
        cur_time = self._get_datetime()
        format_message = self._format_message(message=message, cur_time=cur_time,
                                              message_type=message_type)
        with open(self.logs_path, 'a') as f:
            f.write(format_message)


        return

    def log_params(self, params):
        for param, value in params.items():
            message = f"    {param}:{value}"
            self.log_info(message, message_type="params")

        return

    def _format_message(self, message, cur_time, message_type="info"):
        pre_string = ""
        if(message_type == "new_exp"):
            pre_string = "\n\n\n"
        form_message = f"{pre_string}{cur_time}    {message_type.upper()}: {message}\n"
        return form_message

    def log_arguments(self, args):
        print_("Args:")
        print_("-----")
        for k, v in vars(args).items():
            message = f"  --> {k} = {v}"
            print_(message, message_type="params")
        return

    def _get_datetime(self):
        time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        return time
