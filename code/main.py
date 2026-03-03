from Parser import Parser
from Lemmatizer import Lemmatizer
from LemmatizerHMM import LemmatizerHMM
from TesterLemmatizer import TesterLemmatizer
from sklearn.model_selection import train_test_split
from pathlib import Path

if __name__ == "__main__":
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    data_dir = project_root / "data"
    dict_zip = data_dir / "dict.opcorpora.xml.zip"
    sentences_zip = data_dir / "annot.opcorpora.no_ambig.xml.zip"

    parser_dict = Parser(dict_zip)
    parser_dict.extract_zip_file()
    data_dict = parser_dict.parse_dict_XML_file_cached()
    parser_for_sentences = Parser(sentences_zip)
    parser_for_sentences.extract_zip_file()
    data_sentences = parser_for_sentences.parse_test_XML_file_cached()
    train, test = train_test_split(data_sentences, test_size=0.4, random_state=42)
    lemmatizer = Lemmatizer(data_dict)
    tester_lemmatizer = TesterLemmatizer(lemmatizer)
    tester_lemmatizer.evaluate(test)
    lemmatizer_HMM = LemmatizerHMM(
        data_dict,
        train,
    )
    print(
        *lemmatizer.lemmatize_text(
            "Словил кринж с этого пацана. Он полный нуб, не умеет играть в контру"
        )
    )
    print(
        *lemmatizer.lemmatize_text(
            "Стала стабильнее экономическая и политическая обстановка, предприятия вывели из тени зарплаты сотрудников. Все Гришины одноклассники уже побывали за границей, он был чуть ли не единственным, кого не вывозили никуда дальше Красной Пахры."
        )
    )
    print(len(test))
    tester_lemmatizer_HMM = TesterLemmatizer(lemmatizer_HMM)
    print(
        *lemmatizer_HMM.lemmatize_text(
            "Словил кринж с этого пацана. Он полный нуб, не умеет играть в контру"
        )
    )
    print(
        *lemmatizer_HMM.lemmatize_text(
            "Стала стабильнее экономическая и политическая обстановка, предприятия вывели из тени зарплаты сотрудников. Все Гришины одноклассники уже побывали за границей, он был чуть ли не единственным, кого не вывозили никуда дальше Красной Пахры."
        )
    )
    tester_lemmatizer_HMM.evaluate(test)
