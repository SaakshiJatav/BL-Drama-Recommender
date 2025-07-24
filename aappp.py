import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# -------- Step 1: Load & Clean Data Safely --------
df = pd.read_csv("BL_Drama_Recommendation.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[()]', '', regex=True)

# Ensure Mood_Tags column exists
if 'Mood_Tags' not in df.columns:
    for col in df.columns:
        if 'mood' in col.lower() or 'tags' in col.lower():
            df.rename(columns={col: 'Mood_Tags'}, inplace=True)
            break
    if 'Mood_Tags' not in df.columns:
        df['Mood_Tags'] = 'Not specified'

df['Mood_Tags'] = df['Mood_Tags'].fillna('Not specified')

# Fix rating column name & convert to numeric
for col in df.columns:
    if "rating" in col.lower():
        df.rename(columns={col: "Personal_Rating_out_of_10"}, inplace=True)
        break
if 'Personal_Rating_out_of_10' not in df.columns:
    df['Personal_Rating_out_of_10'] = 0
df['Personal_Rating_out_of_10'] = pd.to_numeric(df['Personal_Rating_out_of_10'], errors='coerce').fillna(0)

# -------- Step 2: Build Recommendation Model --------
df['combined_features'] = df['Summary'] + " " + df['Genres'] + " " + df['Mood_Tags']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def recommend(title, num_recommendations=5):
    if title not in indices:
        return ["Drama not found. Please try again."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    drama_indices = [i[0] for i in sim_scores]
    recommendations = []
    for i in drama_indices:
        rec = f"{df.iloc[i]['Title']} | Genres: {df.iloc[i]['Genres']} | Rating: {df.iloc[i]['Personal_Rating_out_of_10']}/10 | Year: {df.iloc[i]['Year']}"
        recommendations.append(rec)
    return recommendations

def top_rated(num_recommendations=5):
    top = df.sort_values(by='Personal_Rating_out_of_10', ascending=False).head(num_recommendations)
    return [f"{row.Title} | Rating: {row.Personal_Rating_out_of_10}/10 | Year: {row.Year}" for _, row in top.iterrows()]

# -------- Step 3: Streamlit App --------
st.title("BL Drama Recommendation System")
st.write("Type a BL drama title to get recommendations or view the top-rated dramas!")

user_input = st.text_input("Enter a BL drama name:")
if st.button("Recommend"):
    recs = recommend(user_input)
    st.write("### Recommendations:")
    for r in recs:
        st.write(f"- {r}")

if st.button("Show Top Rated"):
    top_recs = top_rated()
    st.write("### Top Rated BL Dramas:")
    for r in top_recs:
        st.write(f"- {r}")