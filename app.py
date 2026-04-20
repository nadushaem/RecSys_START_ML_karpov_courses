import os
from typing import List
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine
from hashlib import md5
from catboost import CatBoostClassifier

from schema import PostGet


# класс, который должен возвращать сервис: группа юзера А или В + список постов
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

# показываем топ 5 постов, если юзер до этого ничего не лайкал
engine = create_engine(
    "postgresql://login:password@"
    "postgres.lab.karpov.courses:6432/startml"
    )

with engine.connect() as connection:
    posts_5 = pd.read_sql("""
        SELECT 
            public.post_text_df.post_id AS id,
            public.post_text_df.text,
            public.post_text_df.topic,
            COUNT(public.feed_data.post_id) AS like_count
        FROM public.post_text_df
        JOIN public.feed_data ON public.feed_data.post_id = public.post_text_df.post_id
        WHERE public.feed_data.action = 'like'
        GROUP BY 
            public.post_text_df.post_id,
            public.post_text_df.text,
            public.post_text_df.topic
        ORDER BY COUNT(public.feed_data.post_id) DESC
        LIMIT 5
    """
, con=connection)

posts_dict = posts_5.drop('like_count', axis=1).to_dict('records')

posts_list = [
    PostGet(
        id=item['id'],
        text=item['text'],
        topic=item['topic']
    )
    for item in posts_dict
]

# загружаем всю информацию о лайках, чтобы фильтровать уже лайкнутые пользователем посты
# число 708969 - это заранее посчитанное количество лайков среди 6кк строк, на которых обучалась модель
# такой лимит поставлен, чтобы не перегружать сервис
with engine.connect() as connection:
    liked_posts_df = pd.read_sql("""
        SELECT user_id, post_id, timestamp FROM public.feed_data WHERE action = 'like'
        ORDER BY timestamp DESC LIMIT 708969""",
        con=connection
    ).drop('timestamp', axis=1)


# определение группы на основе md5 хеша
def get_group(user_id, salt="exp_2026_04_11", ratio=0.5):

    # cоединяем айди пользователя с солью и хешируем
    full_string = f"{user_id}{salt}"
    hash_value = md5(full_string.encode()).hexdigest()

    # берем первые 10 символов хеша и переводим в число (основание 16),
    # затем берем остаток от деления на 100, чтобы получить число от 0 до 99
    hash_int = int(hash_value[:10], 16)

    return 'test' if (hash_int % 100) < (ratio * 100) else 'control'


# функция для деления пользователей на группы:
def test_and_control_groups():
    with engine.connect() as connection:
        user_id_df = pd.read_sql("""
            SELECT DISTINCT user_id FROM public.user_data""",
            con=connection
        ).drop('timestamp', axis=1)
    user_id_df['group'] = user_id_df['user_id'].apply(get_group)

    return user_id_df


# класс нейросети
class SimpleRecMLP(nn.Module):
    def __init__(self, cat_cardinalities, num_user, num_item, user_cat_cols, item_cat_cols, dropout=0.25):
        super().__init__()
        self.user_cat_cols = user_cat_cols
        self.item_cat_cols = item_cat_cols
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col, card in cat_cardinalities.items():
            emb_dim = min(50, max(4, int(round(card ** 0.25 * 8))))
            self.embeddings[col] = nn.Embedding(card, emb_dim)
            nn.init.normal_(self.embeddings[col].weight, std=0.01)
            total_emb_dim += emb_dim

        input_dim = total_emb_dim + num_user + num_item
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, user_cat, user_num, item_cat, item_num):
        embs = []
        for i, col in enumerate(self.user_cat_cols):
            embs.append(self.embeddings[col](user_cat[:, i]))
        for i, col in enumerate(self.item_cat_cols):
            embs.append(self.embeddings[col](item_cat[:, i]))
        x = torch.cat(embs + [user_num, item_num], dim=1)
        return self.mlp(x).squeeze(1)


# загрузка моделей
def get_model_path(path: str, model_type: str = "test") -> str:
    if os.environ.get("IS_LMS") == "1":
        if model_type == "test":
            MODEL_PATH = '/workdir/user_input/model_test'
        else:
            MODEL_PATH = '/workdir/user_input/model_control'
    else:
        MODEL_PATH = path
    return MODEL_PATH


# загрузка тестовой модели (нейросеть)
def load_test_model():
    base_path = get_model_path("/my/super/path", model_type="test")
    model = torch.load(
        base_path,
        map_location="cpu"
    )
    cols = model["feature_cols"]

    model_MLP = SimpleRecMLP(
        cat_cardinalities=model["cat_cardinalities"],
        num_user=len(cols["user_num_cols"]),
        num_item=len(cols["item_num_cols"]),
        user_cat_cols=cols["user_cat_cols"],
        item_cat_cols=cols["item_cat_cols"],
        dropout=0.25
    )
    model_MLP.load_state_dict(model["model_state_dict"])
    model_MLP.eval()

    artifacts = {
        "cat_cardinalities": model["cat_cardinalities"],
        "cols": cols,
    }
    return model_MLP, artifacts


# загрузка модели-контроля (кэтбуст)
def load_control_model():
    model_path = get_model_path("/my/super/path", model_type="control")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


# считывание чанками из крупных таблиц
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 100000
    engine = create_engine(
        "postgresql://login:password@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# загрузка фичей для нейросети (тест)
def load_features_test() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.nadezhda01em_users_test_1')

# загрузка информации о постах для нейросети (тест)
def load_posts_test() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.nadezhda01em_posts_test_1')

# загрузка фичей для кэтбуста (контроль)
def load_features_control() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.nadezhda01em_users_control_1')

# загрузка информации о постах для кэтбуста (контроль)
def load_posts_control() -> pd.DataFrame:
    return batch_load_sql('SELECT * FROM public.nadezhda01em_posts_control_1')


# вспомогательные функции для работы нейросетевой модели:

# берем конкретного юзера
def build_user_row(user_id: int) -> pd.DataFrame:
    row = features_test.loc[features_test["user_id"] == user_id]
    if row.empty:
        return None
    return row.iloc[[0]].copy()

# получаем таблицу, где для каждого поста дублированы признаки конкретного пользователя
def build_candidate_frame(user_row: pd.DataFrame, posts_df: pd.DataFrame) -> pd.DataFrame:
    cand = posts_df.copy()
    for col in [c for c in user_row.columns if c != "user_id"]:
        cand[col] = user_row[col].iloc[0]
    return cand

# приведение к нужным типам данных
def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in art["cols"]["user_cat_cols"] + art["cols"]["item_cat_cols"]:
        df[col] = df[col].astype(int)
    for col in art["cols"]["user_num_cols"] + art["cols"]["item_num_cols"]:
        df[col] = df[col].astype(np.float32)
    return df

# основной шаг инференса
@torch.no_grad()
def predict_scores(df: pd.DataFrame) -> np.ndarray:
    user_cat = torch.tensor(df[art["cols"]["user_cat_cols"]].values, dtype=torch.long)
    user_num = torch.tensor(df[art["cols"]["user_num_cols"]].values, dtype=torch.float32)
    item_cat = torch.tensor(df[art["cols"]["item_cat_cols"]].values, dtype=torch.long)
    item_num = torch.tensor(df[art["cols"]["item_num_cols"]].values, dtype=torch.float32)

    logits = model_test(user_cat, user_num, item_cat, item_num)
    return torch.sigmoid(logits).cpu().numpy()

# рекомендации нейросетевой модели (тест)
def test_group_recommended(user_id: int, limit: int = 5) -> List[PostGet]:
    # получаем данные пользователя
    user_row = build_user_row(user_id)
    if user_row is None:
        # если пользователь не найден, выдаём топ постов из тестовой выборки
        top_posts = posts_test.head(limit)[["post_id", "text", "topic"]].to_dict("records")
        return [PostGet(id=row["post_id"], text=row["text"], topic=row["topic"]) for row in top_posts]

    # фильтруем уже лайкнутые посты
    liked_posts = liked_posts_df.loc[liked_posts_df["user_id"] == user_id, "post_id"].values
    candidate_posts = posts_test[~posts_test["post_id"].isin(liked_posts)].copy()

    if len(candidate_posts) == 0:
        return []

    # подготавливаем данные для модели и получаем предсказания
    candidate_frame = build_candidate_frame(user_row, candidate_posts)
    candidate_frame = prepare_for_model(candidate_frame)
    candidate_frame["score"] = predict_scores(candidate_frame)

    # выбираем top-N постов
    top_posts = (
        candidate_frame.sort_values("score", ascending=False)
        .head(limit)[["post_id", "text", "topic"]]
        .to_dict("records")
    )
    return [PostGet(id=row["post_id"], text=row["text"], topic=row["topic"]) for row in top_posts]

# рекомендации кэтбуста (контроль)
def control_group_recommended(user_id: int, limit: int = 5) -> List[PostGet]:
    # фильтруем уже лайкнутые посты
    liked_posts = liked_posts_df[liked_posts_df["user_id"] == user_id]["post_id"].values

    # если пользователь ни разу ничего не лайкал (loyalty == 0) – выдаём топ-5 популярных постов
    if features_control.loc[features_control["user_id"] == user_id, "loyalty"].iloc[0] == 0:
        return posts_list

    # иначе – применяем модель CatBoost
    user_data = features_control[features_control["user_id"] == user_id]
    candidate_posts = posts_control[~posts_control["post_id"].isin(liked_posts)].copy()

    # добавляем признаки пользователя к каждому кандидату
    user_columns = [col for col in user_data.columns if col != "user_id"]
    for col in user_columns:
        candidate_posts[col] = user_data[col].iloc[0]

    # предсказываем вероятности лайка
    model_features = model_control.feature_names_
    candidate_posts["proba"] = model_control.predict_proba(candidate_posts[model_features])[:, 1]

    # выбираем top-N постов
    top_posts = (
        candidate_posts.sort_values("proba", ascending=False)
        .head(limit)[["post_id", "text", "topic"]]
        .to_dict("records")
    )
    return [PostGet(id=row["post_id"], text=row["text"], topic=row["topic"]) for row in top_posts]

# глобальные переменные для тестовой группы
model_test, art = load_test_model()
features_test = load_features_test()
posts_test = load_posts_test()

# глобальные переменные для контрольной группы
model_control = load_control_model()
features_control = load_features_control()
posts_control = load_posts_control()

app = FastAPI()


# Endpoint GET /post/recommendations/
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 5) -> Response:
    user_id = id
    group = get_group(user_id)

    if group == 'test':
        recommendations = test_group_recommended(user_id, limit)
    else:
        recommendations = control_group_recommended(user_id, limit)

    return Response(exp_group=group, recommendations=recommendations)