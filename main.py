# -*- coding:utf-8 _*-
import os
import shutil
import gradio as gr
from langchain_application import LangChainApplication


# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = '..\\chatglm-6b-noint'  # 本地模型文件 or huggingface远程仓库 THUDM/chatglm-6b
    embedding_model_name = '..\\text2vec-large-chinese'  # 本地模型文件 or huggingface远程仓库 GanymedeNil/text2vec-large-chines
    vector_store_path = './cache'
    docs_path = './docs'
    patterns = ['模型问答', '知识库问答']
    n_gpus=1

config = LangChainCFG()
application = LangChainApplication(config)


if not os.path.exists(config.docs_path):
    os.mkdir(config.docs_path)

file_list = [f for f in os.listdir("docs")]

application.source_service.init_source_vector(file_list)

def upload_file(file):
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    file_list.insert(0, filename)
    application.source_service.add_document("docs/" + filename)

def clear_session():
    return '', None


def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_pattern,
            history=None):
    if history == None:
        history = []


    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input)
        history.append((input, result))
        return '', history, history, ''

    else:
        search_text = '检查相关的文件如下：\n'
        resp = application.get_knowledge_based_answer(
            query=input,
            history_len=1,
            temperature=0.1,
            top_p=0.9,
            top_k=top_k,
            chat_history=history
        )
        history.append((input, resp['result']))
        print(enumerate(resp['source_documents']))
        for idx, source in enumerate(resp['source_documents'][:top_k]):
            filename = source.metadata["filename"]
            search_text += f'{idx}. {filename}\n'
        return '', history, history, search_text

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>LangChain Engine</center></h1>
        <center><font size=3>
        </center></font>
        """)
    state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            embedding_model = gr.Dropdown([
                "text2vec-base"
            ],
                label="Embedding model",
                value="text2vec-base")

            large_language_model = gr.Dropdown(
                [
                    "ChatGLM-6B-int4",
                ],
                label="large language model",
                value="ChatGLM-6B-int4")

            top_k = gr.Slider(1,
                              20,
                              value=2,
                              step=1,
                              label="检索top-k文档",
                              interactive=True)

            use_pattern = gr.Radio(
                [
                    '模型问答',
                    '知识库问答',
                ],
                label="模式",
                value='模型问答',
                interactive=True)


            file = gr.File(label="将文件上传到知识库库，内容要尽量匹配\r\n" \
                            "支持类型为 txt, md, docx, pdf",
                           visible=True,
                           file_types=['.txt', '.md', '.docx', '.pdf']
                           )

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='Chinese-LangChain').style(height=400)
            with gr.Row():
                message = gr.Textbox(label='请输入问题')
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
        with gr.Column(scale=2):
            search = gr.Textbox(label='知识库搜索结果')

        # ============= 触发动作=============
        file.upload(upload_file,
                    inputs=file,
                    outputs=None)
        # 发送按钮 提交
        send.click(predict,
                   inputs=[
                       message,
                       large_language_model,
                       embedding_model,
                       top_k,
                       use_pattern,
                       state
                   ],
                   outputs=[message, chatbot, state, search])

        # 清空历史对话按钮 提交
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        # 输入框 回车
        message.submit(predict,
                       inputs=[
                           message,
                           large_language_model,
                           embedding_model,
                           top_k,
                           use_pattern,
                           state
                       ],
                       outputs=[message, chatbot, state, search])

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    #server_port=8888,
    share=False,
    show_error=True,
    debug=True,
    enable_queue=True,
    inbrowser=True,
)
