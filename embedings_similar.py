from sentence_transformers import SentenceTransformer, util


class EmbedingSimilar:
    def __init__(self):
        self.model = SentenceTransformer(model_name_or_path='paraphrase-MiniLM-L6-v2', local_files_only=False)
        self.model.to("cuda:0")
        self.sentences2 = ['месторождении', 'месторождение', 'АТС', 'РТС', 'Саратоворгсинтез', 'Калиниградморнефть',
                           'Тебойл', 'за',
                           'Лукойл', 'Кубаньэнерго', 'МаринБункер', 'МЦПБ', 'Нижнегороднефтьоргсинтез',
                           'Нижневолжскнефть',
                           'Пермьоргсинтез', 'Ростовэнерго', 'Северозападнефтепродукт', 'Ставропольэнерго',
                           'Уралнефтепрордукт', 'Ухтанефтепереработка', 'Центрнефтепродукт', 'Пермьнефтеоргсинтез',
                           'Когалымнефтегаз', 'Белоярскнефтегаз', 'Лангепаснефтегаз', 'Повхнефтегаз', 'Покачевнефтегаз',
                           'Урайнефтегаз', 'Ямалнефтегаз', 'Югранефтепром', 'Меретояханефтегаз', 'Севернефтегаз',
                           'Усинскнефтегаз', 'Ярёганефтегаз', 'Пермьтотинефть', 'Уралойл', 'Приазовнефть', 'НГДО',
                           'ТПП', 'на', 'в', 'по', 'всем', 'предприятиям', 'НВН', 'НХП', 'ГПЗ', 'УГПЗ', 'АИК', 'ЛИНК',
                           'Инженеринг', 'Ликард', 'Транс', 'Черноморье',
                           'Энергоинженринг', 'Энергосети', 'Энергосервис', 'Экоэнерго', 'Югнефтепродукт', 'ЭЛ5',
                           'СОРС',
                           'ПиС', 'ГиД', 'НПС', 'СИКН', 'ТТП', 'ПОВХ', 'УНПО', 'для',
                           'ННОС', 'ВНП', 'ПНОС', 'УНП', 'НОРСИ', 'НПЗ', 'АВТ', 'ЖУВ', 'НВН', 'КМН', 'ЦДУ ТЭК', 'НПО',
                           'СТЛ', 'ОРУ', 'ОРЭМ', 'РНП', 'газопереработка', 'нефтепродуктопроводы',
                           'Пулытьинское', 'Ухтанефтегаз', 'ЦНП', 'ЮНП',
                           'СЗНП', 'ОБР', 'МДОД', 'ЭЛОУ', 'гос. органы', 'Башнефть-полюс', 'НХП', 'Резервнефтепродукт',
                           'Резервнефтепродукт трейдинг', 'переработки', 'переработка', 'ЛЗС', 'Ритек', 'Ритеке', 'Лукойл', 'Лукойле']
        self.embeddings2 = self.model.encode(self.sentences2)

    def similr_text(self, text_similsr: str) -> str:

        sentences1 = text_similsr.split()
        sentence1_1 = []
        for phrase in sentences1:
            if phrase == phrase.upper():
                sentence1_1.append(phrase)

        embeddings1 = self.model.encode(sentences1)

        def find_highest_similarity(sentence1,embedings1, threshold=0.91):
            highest_similarity_pairs_1 = []
            massive = []

            # Calculate cosine similarity
            cosine_scores = util.pytorch_cos_sim(embedings1, self.embeddings2)
            #print(cosine_scores)


            for idx1, scores in enumerate(cosine_scores):
                max_score = 0
                max_idx2 = -1

                for idx2, score in enumerate(scores):
                    if score > max_score:
                        max_score = score
                        max_idx2 = idx2

                if max_score >= threshold:
                    highest_similarity_pairs_1.append((sentence1[idx1], self.sentences2[max_idx2], max_score.item()))


            return highest_similarity_pairs_1

        # Поиск наибольшего сходства для каждого слова
        highest_similarity_pairs = find_highest_similarity(sentences1,embeddings1)

        # Вывод результатов
        print("Наибольшее семантическое сходство (если больше порога):")
        if highest_similarity_pairs:
            i = 0
            for pair in highest_similarity_pairs:
                print(f"{pair[0]} -> {pair[1]} : {pair[2]:.4f}")
                text_similsr = text_similsr.replace(str(pair[0]), str(pair[1]))

        return text_similsr


text_input = "А какая добыча была жуф в Лукоюлб Перми?" #Надо подумать о подьеме сходства до 0.94
app = EmbedingSimilar()
print(app.similr_text(text_input))