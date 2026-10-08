"""兼容垫片（shim）：为 langchain_community.chat_models 补回被移除的 vertexai 模块。

为什么需要？
-----------
  本项目用 ragas 做生成层评测。ragas 的 ragas/llms/base.py 顶部无条件执行：
      from langchain_community.chat_models.vertexai import ChatVertexAI
  但本项目锁定的 langchain-community==0.4.2 是 **sunset（停止维护）版**，
  已经把 chat_models/vertexai.py 删掉了 → 导入 ragas 时直接 ModuleNotFoundError。

  我们不使用 Google VertexAI，所以只需要一个「能让 import 通过」的占位类即可。

为什么放在这里而不是直接改 site-packages？
  1. 直接改 site-packages 属于「脏补丁」，重装依赖就丢，且不可追溯；
  2. 放在项目内，配合 fix_deps.py 一键应用，别人 clone 后也能复现。

用法
----
  见同目录 fix_deps.py（一条命令自动写入 site-packages）。
"""

# 这个文件的内容会被 fix_deps.py 写进
#   .venv/Lib/site-packages/langchain_community/chat_models/vertexai.py
SHIM_SOURCE = '''"""自动生成的兼容垫片。原模块随 langchain-community 0.4.x 被移除。

仅用于让 ragas 的 import 语句通过；本类不做任何实际功能。
真实使用 VertexAI 请改用 langchain-google-vertexai 独立包。
"""
from langchain_core.language_models.chat_models import BaseChatModel


class ChatVertexAI(BaseChatModel):
    """占位实现：仅为兼容 ragas 的模块级 import。"""

    @property
    def _llm_type(self) -> str:
        return "vertexai-shim"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError(
            "这是兼容垫片，不支持实际调用。需要真实 VertexAI 请安装 langchain-google-vertexai。"
        )
'''
