# from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit.components.v1 as components
import streamlit as st
from lxml import etree
import os
from io import BytesIO
import re
import pandas as pd
import openpyxl
# 设置路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# 判断中文
def tag_lang_is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# 判断选择哪个XSL文件
def get_xsl_file(text):
    return "define2-0-0-cn.xsl" if tag_lang_is_chinese(text) else "define2-0-0-en.xsl"

# 部署翻译模型
@ st.cache_resource
def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def nlp_translate(text, model, tokenizer):
    if not text or not text.strip():
        return ""
    tokenized_input = tokenizer(
        [text],
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 512
         )

    translated_tokens = model.generate(**tokenized_input, max_length=512)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# Streamlit 页面配置
st.markdown("""
    <style>
        h1 {
            font-size: 28px !important;
            color: #004a99;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Define-xml translator", layout="wide")
st.sidebar.title("📘 Define-xml translator")

# 初始化状态
if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = ""

with st.sidebar:

    uploaded_file = st.file_uploader("📂 Upload define.xml", type=["xml"])
    uploaded_excel = st.file_uploader("🧾 Import migrated translation file", type=["xlsx"], key="xlsx_upload")

    # 按tag, orig_text键匹配
    # if uploaded_excel:
    #     df_uploaded = pd.read_excel(uploaded_excel)
    #     for _, row in df_uploaded.iterrows():
    #         tag = row.get("Tag")
    #         orig_text = row.get("Original Text")
    #         translated_text = row.get("Translated Text")
    #         if pd.notna(tag) and pd.notna(orig_text) and pd.notna(translated_text):
    #             st.session_state.translated_dict[(tag, orig_text)] = str(translated_text)
    #     st.success("✅ Translations imported from Excel successfully.")

    # 只按tag键匹配
    if uploaded_excel:
        df_uploaded = pd.read_excel(uploaded_excel)
        for _, row in df_uploaded.iterrows():
            tag = row.get("Tag")
            translated_text = row.get("Translated Text")
            if pd.notna(tag) and pd.notna(translated_text):
                for (key_tag, key_orig_text) in list(st.session_state.translated_dict.keys()):
                    if key_tag == tag:
                        st.session_state.translated_dict[(key_tag, key_orig_text)] = str(translated_text)
        st.success("✅ Translations imported by Tag successfully.")

    if uploaded_file:
        parser = etree.XMLParser(remove_blank_text=True)
        xml_tree = etree.parse(uploaded_file, parser)
        root = xml_tree.getroot()

        # 根据 StudyName 决定 XSL 样式
        sample_text = root.xpath("//*[local-name()='ProtocolName']/text()")
        sample_text = sample_text[0] if sample_text else ""
        xsl_path = os.path.join(base_dir, get_xsl_file(sample_text))

        try:
            with open(xsl_path, "rb") as f:
                xsl_tree = etree.parse(f)
            transform = etree.XSLT(xsl_tree)
            html_result = transform(xml_tree)

            highlight_text = st.session_state.highlight_text

            html_with_js = f"""
            <div id="define_html">{html_result}</div>
            <script>
                function clearHighlights() {{
                    document.querySelectorAll('.highlighted').forEach(el => {{
                        el.classList.remove('highlighted');
                        el.style.backgroundColor = '';
                    }});
                }}
                function highlightText(text) {{
                    clearHighlights();
                    if (!text) return;
                    const matches = [];
                    document.querySelectorAll("*").forEach(el => {{
                        if (el.textContent.trim() === text) {{
                            el.classList.add('highlighted');
                            el.style.backgroundColor = 'yellow';
                            matches.push(el);
                        }}
                    }});
                    if (matches.length > 0) {{
                        matches[0].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}
                }}
                document.addEventListener('DOMContentLoaded', function() {{
                    highlightText("{highlight_text}");
                }});
            </script>
            <style>
                .highlighted {{ background-color: yellow !important; }}
            </style>
            """
            st.components.v1.html(html_with_js, height=800, scrolling=True)
        except Exception as e:
            st.error(f"❌ 样式渲染失败：{e}")


if uploaded_file:
    raw_ns = root.nsmap

    # 将默认命名空间绑定为 'odm'（自定义别名）
    ns = {}
    for prefix, uri in raw_ns.items():
        if prefix is None:
            ns['odm'] = uri
        else:
            ns[prefix] = uri
    ns["xml"] = "http://www.w3.org/XML/1998/namespace"

    nodes = []
    # 1. GlobalVariables
    gv_nodes = root.xpath("//*[local-name()='GlobalVariables']/*")
    for node in gv_nodes:
        if node.text and node.text.strip():
            nodes.append({"tag": etree.QName(node.tag).localname, "text": node.text.strip()})

    # 2. ItemGroupDef: def:Structure, def:Class
    for node in root.xpath("//*[local-name()='ItemGroupDef']"):
        for attr in [f"{{{ns['def']}}}Structure", f"{{{ns['def']}}}Class"]:
            val = node.get(attr)
            if val and val.strip():
                attr_name = attr.split("}")[-1]
                nodes.append({"tag": f"ItemGroupDef@{attr_name}", "text": val.strip()})

        desc_nodes = node.xpath(".//*[local-name()='Description']/*[local-name()='TranslatedText'][not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN']")

        for tt in desc_nodes:
            if tt.text and tt.text.strip():
                nodes.append({"tag": "ItemGroupDef.Description", "text": tt.text.strip()})

    # 3.ItemDef: Description
    for node in root.xpath("//*[local-name()='ItemDef']"):
        tag = node.get("OID")  # 原样保留 OID（如 IT.ADMB.STUDYID）
        # 仅当 OID 含两个.（例如 "IT.ADMB.STUDYID"）时才处理
        if tag.count(".") == 2:
            orig_text = node.xpath(".//*[local-name()='Description']/*[local-name()='TranslatedText'][not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN']")[0].text.strip()
            nodes.append({"tag": tag, "text": orig_text})

        # 4：ItemDef 中 def:Origin
        origin_desc_nodes = node.xpath(
            ".//def:Origin/*[local-name()='Description']/*[local-name()='TranslatedText'][not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN']",
            namespaces=ns
        )
        for desc in origin_desc_nodes:
            if desc.text and desc.text.strip():
                tag = f"{tag}.Origin"  # 原有 tag 是 IT.ADSL.SUBJID，加上后缀区分
                nodes.append({"tag": tag, "text": desc.text.strip()})

    # 5. CommentDefxd
    comment_oids = root.xpath("//*[local-name()='ItemDef'][@def:CommentOID]", namespaces=ns)
    for item in comment_oids:
        comment_oid = item.get(f"{{{ns['def']}}}CommentOID")
        if not comment_oid:
            continue

        comment_def = root.xpath(f"//def:CommentDef[@OID='{comment_oid}']", namespaces=ns)
        if comment_def:
            desc_nodes = comment_def[0].xpath(
                ".//*[local-name()='Description']/*[local-name()='TranslatedText'][not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN']",
                namespaces=ns
            )
            for desc in desc_nodes:
                if desc.text and desc.text.strip():
                    tag = f"CommentDef@{comment_oid}"
                    nodes.append({"tag": tag, "text": desc.text.strip()})

    # 6. MethodDef
    for node in root.xpath("//*[local-name()='MethodDef']"):
        desc_nodes = node.xpath(
            ".//*[local-name()='Description']/*[local-name()='TranslatedText'][not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN']")
        for tt in desc_nodes:
            if tt.text and tt.text.strip():
                tag = node.get('OID')
                nodes.append({"tag": tag, "text": tt.text.strip()})

    # 按tag & text 去重
    seen = set()
    unique_nodes = []
    for n in nodes:
        tag = n["tag"]
        text = n["text"]
        # 构造用于去重的 display_tag（保留最后一个字段）
        if tag.startswith("ItemGroupDef@"):
            display_tag = "[Dataset] " + tag.split("@")[1]
        elif tag.startswith("ItemGroupDef.Description"):
            display_tag = "[Dataset] Description"
        elif tag.startswith("IT") and tag.count(".") == 2:
            display_tag = "[Variable] " + tag.split(".")[-1] + "(Label)"
        elif ".Origin" in tag:
            display_tag = "[Variable] " + tag.split(".")[-2]+ "(Derivation)"
        elif tag.startswith("CommentDef"):
            display_tag = "[Comment] " + ".".join(tag.split(".")[-2:])
        elif tag.startswith("MT."):
            display_tag = "[Method] " + ".".join(tag.split(".")[-2:])
        else:
            display_tag = tag

        key = (display_tag, text)
        if key not in seen:
            seen.add(key)
            unique_nodes.append({**n, "display_tag": display_tag})  # 加上 display_tag

    st.success(f"Found {len(unique_nodes)} items that need to be translated")

    if "translated_dict" not in st.session_state:
        st.session_state.translated_dict = {}

    # 翻译按钮
    # col_trans1, col_trans2 = st.columns(2)
    # with col_trans1:
    #     if st.button("🌐English → 中文"):
    #         for node in unique_nodes:
    #             key = (node["tag"], node["text"])
    #             try:
    #                 st.session_state.translated_dict[key] = GoogleTranslator(source='en', target='zh-CN').translate(node["text"])
    #             except:
    #                 st.warning(f"Failed: {node['text']}")
    #
    # with col_trans2:
    #     if st.button("🌐中文 → English"):
    #         for node in unique_nodes:
    #             key = (node["tag"], node["text"])
    #             try:
    #                 st.session_state.translated_dict[key] = GoogleTranslator(source='zh-CN', target='en').translate(node["text"])
    #             except:
    #                 st.warning(f"Failed: {node['text']}")
    col1, col2 = st.columns([1,1])
    with col1:
        MODEL_MAP = {
            "English -> 中文": "TencentARC/opus-mt-en-zh-med",
            "中文 -> English": "Helsinki-NLP/opus-mt-zh-en",
        }
        selected_direction = st.selectbox("", list(MODEL_MAP.keys()), label_visibility="collapsed")
        model_name = MODEL_MAP[selected_direction]

        model, tokenizer = load_model(model_name)
        st.success(f"model '{model_name}' loaded successfully")

    with col2:
        if st.button("🌐 Translate"):
            progress_bar = st.progress(0, text="Translating...")
            total = len(unique_nodes)
            success_count = 0

            for i, node in enumerate(unique_nodes):
                key = (node["tag"], node["text"])
                try:
                    if key not in st.session_state.translated_dict or not st.session_state.translated_dict[key].strip():
                        # translated = GoogleTranslator(source=source_lang,target=target_lang).translate(node["text"])
                        translated = nlp_translate(node["text"],model, tokenizer)
                        st.session_state.translated_dict[key] = translated
                        success_count += 1
                except Exception as e:
                    st.warning(f"❌ Failed: {node['text']} → {e}")
                progress_bar.progress((i + 1) / total, text=f"Translating... ({i + 1}/{total})")

            progress_bar.empty()
            st.success(f"✅ Finished: {success_count}/{total} translated successfully.")

    # 编辑区域
    for i, node in enumerate(unique_nodes):
        tag, orig_text = node["tag"], node["text"]
        display_tag = node.get("display_tag", tag)  # 优先使用去重阶段的 display_tag
        key = (tag, orig_text)  # 注意，这里翻译还是用原始 tag 做键
        default_value = st.session_state.translated_dict.get(key, "")

        def highlight_callback(text=orig_text):
            st.session_state.highlight_text = text

        with st.expander(f"**[{i + 1}] {display_tag}:** {orig_text}"):
            translated_input = st.text_area("", value=default_value, key=f"translated_{i}", height=100)
            st.session_state.translated_dict[key] = translated_input
            st.button("🔗 Goto", on_click=highlight_callback, args=(orig_text,), key=f"btn_{i}")

    # 保存
    if st.button("💾 Save the translated items"):
        export_data = []
        for node in unique_nodes:
            tag = node["tag"]
            orig_text = node["text"]
            display_tag = node.get("display_tag", tag)
            translated_text = st.session_state.translated_dict.get((tag, orig_text), "")
            export_data.append({
                "Display Tag": display_tag,
                "Tag": tag,
                "Original Text": orig_text,
                "Translated Text": translated_text
            })
        df_export = pd.DataFrame(export_data)
        towrite = BytesIO()
        df_export.to_excel(towrite, index=False)
        towrite.seek(0)

        count_updated = 0
        changes = []

        for (tag, orig_text), translated_text in st.session_state.translated_dict.items():
            if tag.startswith("ItemGroupDef@"):
                tag_name, attr_name = tag.split('@')
                is_def_attr = attr_name in ["Structure", "Class"]
                attr_qname = f"{{{ns['def']}}}{attr_name}" if is_def_attr else attr_name
                xpath_expr = f"//*[local-name()='{tag_name}' and @{('def:' if is_def_attr else '') + attr_name}='{orig_text}']"
                nodes_with_attr = root.xpath(xpath_expr, namespaces=ns if is_def_attr else None)
                for elem in nodes_with_attr:
                    old_val = elem.get(attr_qname)
                    elem.set(attr_qname, translated_text)
                    changes.append({"tag": tag, "from": old_val, "to": translated_text})
                    count_updated += 1

            elif tag.startswith("ItemGroupDef.Description") or (tag.startswith("IT") and tag.count(".") == 2):
                desc_nodes = root.xpath(
                    f"""
                    (
                        //*[local-name()='ItemDef']//*[local-name()='Description']/*[local-name()='TranslatedText']
                        |
                        //*[local-name()='ItemGroupDef']//*[local-name()='Description']/*[local-name()='TranslatedText']
                        |
                        //*[local-name()='CommentDef']//*[local-name()='Description']/*[local-name()='TranslatedText']
                    )
                    [(not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN') and normalize-space(text())="{orig_text}"]
                    """,
                    namespaces=ns
                )

                for en_node in desc_nodes:
                    parent = en_node.getparent()
                    old_texts = [c.text for c in parent.xpath("./*[local-name()='TranslatedText']")]
                    for c in parent.xpath("./*[local-name()='TranslatedText']"):
                        parent.remove(c)
                    new_node = etree.Element("TranslatedText")
                    new_node.set(f"{{{ns['xml']}}}lang", "zh-CN" if tag_lang_is_chinese(translated_text) else "en")
                    new_node.text = translated_text
                    parent.append(new_node)
                    changes.append({"tag": tag, "from": "; ".join(old_texts), "to": translated_text})
                    count_updated += 1

            elif ".Origin" in tag:
                base_oid = tag.rsplit(".Origin", 1)[0]
                origin_nodes = root.xpath(
                    f"""
                    //*[local-name()='ItemDef'][@OID='{base_oid}']
                    /*[local-name()='Origin']
                    /*[local-name()='Description']
                    /*[local-name()='TranslatedText']
                    [ (not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN') and normalize-space(text())="{orig_text}" ]
                    """,
                    namespaces=ns
                )
                for en_node in origin_nodes:
                    parent = en_node.getparent()
                    for c in parent.xpath("./*[local-name()='TranslatedText']"):
                        parent.remove(c)
                    new_node = etree.Element("TranslatedText")
                    new_node.set(f"{{{ns['xml']}}}lang", "zh-CN" if tag_lang_is_chinese(translated_text) else "en")
                    new_node.text = translated_text
                    parent.append(new_node)
                    changes.append({"tag": tag, "from": orig_text, "to": translated_text})
                    count_updated += 1

            elif tag.startswith("CommentDef@"):
                comment_oid = tag.split("@")[1]
                desc_nodes = root.xpath(
                    f"""
                    //*[local-name()='CommentDef'][@OID='{comment_oid}']
                    /*[local-name()='Description']/*[local-name()='TranslatedText']
                    [(not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN') and normalize-space(text())="{orig_text}"]
                    """,
                    namespaces=ns
                )
                for en_node in desc_nodes:
                    parent = en_node.getparent()
                    old_texts = [c.text for c in parent.xpath("./*[local-name()='TranslatedText']")]
                    for c in parent.xpath("./*[local-name()='TranslatedText']"):
                        parent.remove(c)
                    new_node = etree.Element("TranslatedText")
                    new_node.set(f"{{{ns['xml']}}}lang", "zh-CN" if tag_lang_is_chinese(translated_text) else "en")
                    new_node.text = translated_text
                    parent.append(new_node)
                    changes.append({"tag": tag, "from": "; ".join(old_texts), "to": translated_text})
                    count_updated += 1

            elif tag.startswith("MT."):
                desc_nodes = root.xpath(
                    f"""
                        //*[local-name()='MethodDef'][@OID='{tag}']
                        /*[local-name()='Description']/*[local-name()='TranslatedText']
                        [(not(@xml:lang) or @xml:lang='en' or @xml:lang='zh-CN') and normalize-space(text())="{orig_text}"]
                        """,
                    namespaces=ns
                )

                for en_node in desc_nodes:
                    parent = en_node.getparent()
                    if parent is not None:
                        old_texts = [c.text for c in parent.xpath("./*[local-name()='TranslatedText']")]
                        parent = en_node.getparent()
                        old_texts = [c.text for c in parent.xpath("./*[local-name()='TranslatedText']")]
                        for c in parent.xpath("./*[local-name()='TranslatedText']"):
                            parent.remove(c)
                        new_node = etree.Element("TranslatedText")
                        new_node.set(f"{{{ns['xml']}}}lang", "zh-CN" if tag_lang_is_chinese(translated_text) else "en")
                        new_node.text = translated_text
                        parent.append(new_node)
                        changes.append({"tag": tag, "from": "; ".join(old_texts), "to": translated_text})
                        count_updated += 1

            else:
                matches = root.xpath(f"//*[local-name()='{tag}' and normalize-space(text())='{orig_text}']")
                for match in matches:
                    old_val = match.text
                    match.text = translated_text
                    changes.append({"tag": tag, "from": old_val, "to": translated_text})
                    count_updated += 1

        out = BytesIO()
        xml_tree.write(out, encoding="utf-8", xml_declaration=True, pretty_print=True)

        if changes:
            st.markdown("### 📝Translated Log")
            for i, c in enumerate(changes, 1):
                st.markdown(f"**[{i}] {c['tag']}**: `{c['from']}` → ✅ `{c['to']}`")

        st.success(f"✅ Successfully replaced {count_updated} items.")

        st.session_state.saved_excel = towrite.getvalue()
        st.session_state.saved_xml = out.getvalue()
        st.session_state.saved = True

    if st.session_state.get("saved", False):
        st.markdown("### 📥 Download Translated Files")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download define_translation.xlsx",
                               st.session_state.saved_excel,
                               file_name="define_translation.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col2:
            st.download_button("📥 Download define_translated.xml",
                               st.session_state.saved_xml,
                               file_name="define_translated.xml",
                               mime="application/xml")
