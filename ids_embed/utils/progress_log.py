import logging
import traceback
import threading

import enlighten

CATE_Parse_IDS = "Parse_IDS"
CATE_Parse_IDS_cache = "Parse_IDS cached reload"
CATE_Grounding = "Grounding Glyph"
CATE_Grounding_cache = "Grounded Shape cached reload"
CATE_Construct_component = "Construct component"
CATE_Synthesis = "Synthesis"

CATE_LIST = [
    CATE_Parse_IDS,
    CATE_Parse_IDS_cache,
    CATE_Grounding,
    CATE_Grounding_cache,
    CATE_Construct_component,
    CATE_Synthesis,
]


class ProgressManager:
    def __init__(self):
        self.enlighten_mgr = enlighten.get_manager()
        self.loggers = {}
        self.status_bar = self.enlighten_mgr.status_bar(
            status_format="ids_embed currently at: {stage}{fill}{elapsed}",
            color="bold_underline_bright_white_on_lightslategray",
            justify=enlighten.Justify.CENTER,
            stage="Initializing",
            autorefresh=True,
            min_delta=0.0001,
        )
        CATE_LIST.copy()

        self.call_stack = []
        # pop num is used for delayed execution of pop stage
        self.pop_num = 0

    def get_stage(self):
        return " | ".join(self.call_stack)

    def actual_pop(self):
        while self.pop_num > 0:
            self.call_stack.pop()
            self.pop_num -= 1

    def actual_pop_and_update(self):
        self.actual_pop()
        self.update_stage_bar()

    def push_stage(self, text):
        self.actual_pop()
        if hasattr(self, "pop_delay_timier") and self.pop_delay_timier is not None:
            self.pop_delay_timier.cancel()
            self.pop_delay_timier = None
        self.call_stack.append(text)

    def pop_stage(self):
        # for better visualization (not hurting your eyes), pop is delay executed.
        # either by 0.1 seconds, or it is executed till next push happen
        self.pop_num += 1
        self.pop_delay_timier = threading.Timer(0.1, self.actual_pop_and_update)
        self.pop_delay_timier.start()

    def update_stage_bar(self):
        mgr.status_bar.update(stage=mgr.get_stage())


def update_push_stage(text):
    mgr.push_stage(text)
    mgr.update_stage_bar()


def update_pop_stage():
    mgr.pop_stage()
    mgr.update_stage_bar()


def update_total_task(cate_name, total=1):
    if cate_name not in mgr.loggers:
        mgr.loggers[cate_name] = mgr.enlighten_mgr.counter(total=total, desc=cate_name)
    logger = mgr.loggers[cate_name]
    logger.update(incr=0, force=True)


def update_completed_task(cate_name, inc=1):
    logger = mgr.loggers[cate_name]
    logger.update(incr=inc)


def progress_log_update(description):
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            update_push_stage("{} ({})".format(description, args[0]))
            ret = func(*args, **kwargs)
            update_pop_stage()
            return ret

        return wrapper_func

    return decorator_func


mgr = ProgressManager()
