from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask_cors import CORS
from flask import Flask, request
from flask_pymongo import PyMongo, ObjectId
from bson.json_util import dumps
# pass: wLSbF6v2xJT10lUds

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = "mongodb+srv://jash-2000:wLSbF6v2xJT10lUd@lab.1ufwyzv.mongodb.net/SkinCare?retryWrites=true&w=majority"
mongo = PyMongo(app)

vectorizer = TfidfVectorizer()

@app.route('/user-questions/<user_email>', methods=['GET'])
def get_user_questions(user_email):
    questions = mongo.db.recommendations.find({"email":user_email})
    return dumps(questions), 200


@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    user_email = request.json['user_email']
    user_description = request.json['question']

    engine_data = mongo.db.engine_products.find()
    df_engine = pd.DataFrame(engine_data)

    description_vectors = vectorizer.fit_transform(df_engine['both'])
    user_description_vector = vectorizer.transform([user_description])

    similarity_scores = cosine_similarity(user_description_vector, description_vectors).flatten()

    df_engine['similarity_score'] = similarity_scores
    df_sorted = df_engine.sort_values(by='similarity_score', ascending=False)

    recommended_products = df_sorted['_id'].head(10 ).tolist()

    product_docs = [mongo.db.product.find_one({"_id": product_id}) for product_id in recommended_products]

    id = mongo.db.recommendations.insert_one({
        'email': user_email,
        'question': user_description,
        'recommended_products': recommended_products
    }).inserted_id

    return dumps({"id": str(id), "products": product_docs}), 200

@app.route('/recommendation/<question_id>', methods=['GET'])
def get_products_by_question(question_id):
    question_doc = mongo.db.recommendations.find_one({"_id": ObjectId(question_id)})
    product_ids = question_doc['recommended_products']
    product_docs = [mongo.db.product.find_one({"_id": ObjectId(product_id)}) for product_id in product_ids]
    return dumps(product_docs), 200

if __name__ == "__main__":
    app.run(debug=True)