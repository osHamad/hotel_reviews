from csv import reader
from sklearn.feature_extraction.text import TfidfVectorizer


def to_list(data, pos_num):
    word_list = []
    for atr in data:
        word_list.append(atr[pos_num])

    return word_list


def main():
    with open('tripadvisor_hotel_reviews.csv', newline='', encoding='utf-8') as hotel_data:
        hotel_data = list(reader(hotel_data))
        hotel_data.pop(0)

    review = to_list(hotel_data, 0)
    rating = to_list(hotel_data, 1)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    vectorizer.fit_transform(review, rating)


if __name__ == '__main__':
    main()