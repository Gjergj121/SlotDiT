import torch
import torch.nn as nn

import re
import html
import urllib.parse as ul
import ftfy
from bs4 import BeautifulSoup

from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

class T5TextEmbedder(nn.Module):
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')

    def __init__(self, device, model_name="t5-small", model_max_length=120, use_text_preprocessing=True):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(model_name).eval()
        self.model_max_length = model_max_length
        self.use_text_preprocessing = use_text_preprocessing

    def embed_texts(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]

        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            text_encoder_embs = self.text_encoder(
                input_ids=text_tokens_and_mask['input_ids'].to(self.device),
                attention_mask=text_tokens_and_mask['attention_mask'].to(self.device),
            )['last_hidden_state'].detach()
        return text_encoder_embs, text_tokens_and_mask['attention_mask'].to(self.device)

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)

        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)

        caption = BeautifulSoup(caption, features='html.parser').text

        caption = re.sub(r'@[\w\d]+\b', '', caption)

        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)

        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            '-', caption)

        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        caption = re.sub(r'&quot;?', '', caption)

        caption = re.sub(r'&amp', '', caption)

        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        caption = re.sub(r'\\n', ' ', caption)

        caption = re.sub(r'#\d{1,3}\b', '', caption)

        caption = re.sub(r'#\d{5,}\b', '', caption)

        caption = re.sub(r'\b\d{6,}\b', '', caption)

        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        caption = re.sub(r'[\"\']{2,}', r'"', caption)
        caption = re.sub(r'[\.]{2,}', r' ', caption)

        caption = re.sub(self.bad_punct_regex, r' ', caption)
        caption = re.sub(r'\s+\.\s+', r' ', caption)

        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()
