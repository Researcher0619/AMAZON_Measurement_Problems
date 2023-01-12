import pandas as pd
import math
import scipy.stats as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', 500)
pd.set_option("display.width", 500)
amazon_data =pd.read_csv("Datasets/sltk_amazon.csv")
amazon_data_copy = amazon_data.copy()
amazon_data.head()

#Task 1: Calculate the Average Rating according to the current comments and compare it with the existing average rating.


amazon_data.describe().T
amazon_data.isnull().sum()
amazon_data.shape
amazon_data["asin"].unique()
amazon_data['overall'].value_counts()
existing_average_rating = amazon_data["overall"].mean()
existing_average_rating




# Time-Based Weighted Average Rating


amazon_data["unixReviewTime"] = pd.to_datetime(amazon_data["unixReviewTime"])
amazon_data["reviewTime"] = pd.to_datetime(amazon_data["reviewTime"])
amazon_data["day_diff"] = pd.to_datetime(amazon_data["day_diff"])
amazon_data.info()
amazon_data.head()

amazon_data.shape

current_date = amazon_data["reviewTime"].max()
current_date
amazon_data["days"] = (current_date - amazon_data["reviewTime"]).dt.days
amazon_data["days"].head()
amazon_data["days"].sort_values(ascending=False).head()


amazon_data.head()

amazon_data["days"].quantile([0.25,0.5,0.75,1]).T
amazon_data["days"].describe().T

existing_average_rating
amazon_data.loc[amazon_data["days"] <= 280, "overall"].mean()
amazon_data.loc[(amazon_data["days"] > 280) & (amazon_data["days"] <= 430), "overall"].mean()
amazon_data.loc[(amazon_data["days"] > 430) & (amazon_data["days"] <= 600), "overall"].mean()
amazon_data.loc[amazon_data["days"] > 600, "overall"].mean()


Time_Based_Weighted_Average_Rating = amazon_data.loc[amazon_data["days"] <= 280, "overall"].mean()* 28/100 + \
amazon_data.loc[(amazon_data["days"] > 280) & (amazon_data["days"] <= 430), "overall"].mean() * 26/100 + \
amazon_data.loc[(amazon_data["days"] > 430) & (amazon_data["days"] <= 600), "overall"].mean() * 24/100 + \
amazon_data.loc[amazon_data["days"] > 600, "overall"].mean() * 22/100

print(Time_Based_Weighted_Average_Rating, existing_average_rating)

# Time_Based_Weighted_Average_Rating and existing_average_rating are almost the same.

def time_based_weighted_avarage (Dataframe, w1= 28, w2= 26, w3= 24, w4= 22):
    Dataframe["days"] = (current_date - Dataframe["reviewTime"]).dt.days
    Dataframe["days"].sort_values(ascending=False).head()
    Dataframe["days"].quantile([0.25,0.5,0.75,1]).T
    Dataframe["days"].describe().T
    Time_Based_Weighted_Average_Rating = Dataframe.loc[Dataframe["days"] <= 280, "overall"].mean()* w1/100 + \
    Dataframe.loc[(Dataframe["days"] > 280) & (Dataframe["days"] <= 430), "overall"].mean() * w2/100 + \
    Dataframe.loc[(Dataframe["days"] > 430) & (Dataframe["days"] <= 600), "overall"].mean() * w3/100 + \
    Dataframe.loc[Dataframe["days"] > 600, "overall"].mean() * w4/100
    return Time_Based_Weighted_Average_Rating


# User-Based Weighted Average Rating:

amazon_data["helpful_no"] = amazon_data["total_vote"] - amazon_data["helpful_yes"]
amazon_data["helpful_no"].sort_values(ascending=False)

amazon_data.sort_values("total_vote", ascending=False)
amazon_data.sort_values("helpful", ascending=False)
amazon_data["helpful_yes"].sort_values(ascending=False)
amazon_data["overall"].sort_values(ascending=False)
amazon_data["days"].sort_values(ascending=False)

def score_up_down_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

def score_average_rating(helpful_yes,helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    helpful_yes: int
        up count
    helpful_no: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


comments = pd.DataFrame({"helpful_yes": amazon_data["helpful_yes"], "helpful_no": amazon_data["helpful_no"]})

comments.sort_values("helpful_yes", ascending=False)

comments
# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

fancy = ["score_pos_neg_diff", "score_average_rating", "wilson_lower_bound","helpful_yes", "helpful_no"]
comments = comments.sort_values(fancy, ascending=False)
comments = comments.sort_values("wilson_lower_bound", ascending=False)
