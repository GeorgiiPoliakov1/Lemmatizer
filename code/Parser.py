import zipfile as zf
import os
import hashlib
import pickle
from lxml import etree
import pandas as pd
import re
from pathlib import Path
import string


class Parser:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.zip_dir_name = os.path.dirname(zip_path) or "."
        self.chars_to_remove = "".join(
            set(string.punctuation + '«»"„“‘’()[]{};:+-*/=@#$%^&_|•—–—↑↓→←…\©®™§°±')
        )
        self.base_name = Path(zip_path).stem
        self.pkl_path = os.path.join(self.zip_dir_name, f"{self.base_name}_cached.pkl")
        self.test_pkl_path = os.path.join(
            self.zip_dir_name, f"{self.base_name}_test_cached.pkl"
        )
        self.xml_path = None
        self.zip_hash_path = os.path.join(
            self.zip_dir_name, f"{self.base_name}_zip_hash.txt"
        )

    def _get_zip_hash(self):
        hash_md5 = hashlib.md5()
        with open(self.zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_zip_hash(self):
        if os.path.exists(self.zip_hash_path):
            with open(self.zip_hash_path, "r") as f:
                return f.read().strip()
        return None

    def _save_zip_hash(self, zip_hash):
        with open(self.zip_hash_path, "w") as f:
            f.write(zip_hash)

    def _clean(self, chars_to_remove: str, text: str) -> str:
        if not text:
            return ""
        regex = re.compile(f"[{re.escape(chars_to_remove)}]")
        return regex.sub("", text).strip()

    def extract_zip_file(self, extract_path=None):
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"ZIP-file wasn't found: {self.zip_path}")
        if extract_path is None:
            extract_path = self.zip_dir_name
        with zf.ZipFile(self.zip_path) as zip_file:
            xml_files = [f for f in zip_file.namelist() if f.endswith(".xml")]
            if not xml_files:
                raise ValueError("No XML files found in the archive.")
            xml_filename = xml_files[0]
            self.xml_path = os.path.join(extract_path, xml_filename)
            if not os.path.exists(self.xml_path):
                zip_file.extractall(extract_path)
                print(f"Успешно распаковано: {self.xml_path}")
            else:
                print(f"XML файл уже существует: {self.xml_path}")
        return self.xml_path

    def parse_test_XML_file(
        self,
        xml_path: str = None,
        chars_to_remove: str = None,
    ):
        if chars_to_remove is None:
            chars_to_remove = self.chars_to_remove
        if xml_path is None:
            xml_path = self.xml_path
        if xml_path is None or not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        data = []
        context = etree.iterparse(xml_path, events=("end",), tag="sentence")
        print("Начинается загрузка данных для тестирования...")
        count = 0

        for event, sentence in context:
            source_node = sentence.find("source")
            tokens_node = sentence.find("tokens")

            if source_node is not None and tokens_node is not None:
                tokens_list = []

                for token in tokens_node.findall("token"):
                    raw_word = token.get("text")
                    clean_word = self._clean(chars_to_remove, raw_word)

                    if not clean_word:
                        continue

                    v_node = token.find(".//v")
                    if v_node is not None:
                        l_node = v_node.find("l")
                        g_node = l_node.find("g") if l_node is not None else None

                        lemma = (
                            l_node.get("t")
                            if l_node is not None
                            else clean_word.lower()
                        )
                        pos = g_node.get("v") if g_node is not None else "UNKN"

                        tokens_list.append(
                            {
                                "Слово": clean_word,
                                "Исходное слово": lemma.lower(),
                                "Часть речи": pos,
                            }
                        )

                clean_source = self._clean(self.chars_to_remove, source_node.text)
                data.append({"Исходный текст": clean_source, "токены": tokens_list})
                count += 1

                if count % 1000 == 0:
                    print(f"Обработано предложений: {count}")

            sentence.clear()
            while sentence.getprevious() is not None:
                parent = sentence.getparent()
                if parent is not None:
                    parent.remove(sentence)
        print(f"Загрузка завершена. Всего предложений: {len(data)}")
        df = pd.DataFrame(data)
        print(f"Сохранение тестовых данных в кэш: {self.test_pkl_path}...")
        try:
            with open(self.test_pkl_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._save_zip_hash(self._get_zip_hash())
            print("Тестовые данные успешно сохранены в кэш.")
        except Exception as e:
            print(f"Ошибка при сохранении кэша: {e}")

        return df

    def parse_test_XML_file_cached(
        self,
        xml_path: str = None,
        chars_to_remove: str = None,
        force_reparse: bool = False,
    ):
        if xml_path is None:
            if self.xml_path is None or not os.path.exists(self.xml_path):
                self.extract_zip_file()
            xml_path = self.xml_path

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        if not force_reparse and os.path.exists(self.test_pkl_path):
            current_zip_hash = self._get_zip_hash()
            saved_zip_hash = self._load_zip_hash()

            if current_zip_hash == saved_zip_hash:
                print(f"Загрузка тестовых данных из кэша: {self.test_pkl_path}")
                try:
                    with open(self.test_pkl_path, "rb") as f:
                        df = pickle.load(f)
                    print(f"Успешно загружено {len(df)} предложений.")
                    return df
                except Exception as e:
                    print(
                        f"Ошибка при чтении кэша ({e}), выполняется повторный парсинг."
                    )
            else:
                print(
                    "ZIP-файл изменён, кэш тестовых данных устарел. Выполняем парсинг."
                )
        else:
            if force_reparse:
                print("Принудительный режим: игнорируется кэш тестовых данных.")
            else:
                print(
                    f"Кэш тестовых данных не найден ({self.test_pkl_path}). Выполняется парсинг."
                )
        self.parse_test_XML_file(chars_to_remove)

    def parse_dict_XML_file(self, xml_file_path=None, chars_to_remove: str = None):
        if xml_file_path is None:
            xml_file_path = self.xml_path
        if xml_file_path is None or not os.path.exists(xml_file_path):
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")
        if chars_to_remove is None:
            chars_to_remove = self.chars_to_remove
        regex = re.compile(f"[{re.escape(self.chars_to_remove)}]")
        words, initial_forms, poses = [], [], []
        context = etree.iterparse(xml_file_path, events=("end",), tag="lemma")
        print("Начинается парсинг XML")
        count = 0
        for event, elem in context:
            l_tag = elem.find("l")
            if l_tag is None:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                continue
            initial_form = l_tag.get("t")
            if not initial_form:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                continue
            g_tag = l_tag.find("g")
            pos = g_tag.get("v") if g_tag is not None else "UNKN"
            f_tags = elem.findall("f")
            if f_tags:
                for f_tag in f_tags:
                    word = f_tag.get("t")
                    if not word:
                        continue
                    word_lower = word.lower()
                    if not regex.search(word_lower):
                        words.append(word_lower)
                        initial_forms.append(initial_form)
                        poses.append(pos)
                        count += 1
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            if count % 100000 == 0:
                print(f"Обработано форм: {count}")
        print(f"Парсинг завершен. Всего записей: {len(words)}")
        return pd.DataFrame(
            {"Слово": words, "Исходное слово": initial_forms, "Часть речи": poses}
        )

    def parse_dict_XML_file_cached(self, force_reparse=False):
        if self.xml_path is None or not os.path.exists(self.xml_path):
            self.extract_zip_file()
        if not force_reparse and os.path.exists(self.pkl_path):
            current_zip_hash = self._get_zip_hash()
            saved_zip_hash = self._load_zip_hash()
            if current_zip_hash == saved_zip_hash:
                print(f"Загрузка готовых данных из кэша: {self.pkl_path}")
                try:
                    with open(self.pkl_path, "rb") as f:
                        df = pickle.load(f)
                    print(f"Успешно загружено {len(df)} записей.")
                    return df
                except Exception as e:
                    print(
                        f"Ошибка при чтении кэша ({e}), выполняется повторный парсинг."
                    )
            else:
                print("ZIP-файл изменён, кэш устарел. Выполняется парсинг.")
        else:
            if force_reparse:
                print("Принудительный режим: игнорируем кэш.")
            else:
                print(f"Кэш не найден ({self.pkl_path}). Выполняется парсинг.")
        df = self.parse_dict_XML_file()
        print(f"Сохранение результата в {self.pkl_path}.")
        with open(self.pkl_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._save_zip_hash(self._get_zip_hash())
        print("Данные успешно сохранены в кэш.")
        return df
