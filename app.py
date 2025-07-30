from flask import Flask, jsonify, request
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import joblib
import time
from threading import Lock

app = Flask(__name__)

lock = Lock()

data_path = "predictor_file.csv"
fighter_details_path = "fighter_details.csv"
ufc_data_path = "UFC.csv"
model_path = "logistic_model.joblib"

model = joblib.load(model_path)

data = pd.read_csv(data_path)
df_fighter = pd.read_csv(fighter_details_path)
df_ufc = pd.read_csv(ufc_data_path)

df_events = df_ufc[['event_id', 'event_name', 'location', 'date']].copy()
df_events.drop_duplicates(inplace=True)
df_events = df_events.sort_values(by='date', ascending=False)

r_df = data.add_prefix('r_')
b_df = data.add_prefix('b_')

def create_session():
    session = requests.Session()
    retry_strat = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET']
    )
    adapter = HTTPAdapter(max_retries=retry_strat)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_session()

def get_event_link(session):
    url = "http://ufcstats.com/statistics/events/completed"
    while True:
        try:
            response = session.get(url, headers = {'User-Agent' : UserAgent().random})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            tag = soup.find('a', class_="b-link b-link_style_white")
            print(f"Link Extracted {tag['href']}")
            return tag['href'], tag.text.strip()
        except:
            time.sleep(60)

def get_fight_links(session, event_url):
    while True:
        try:
            response = session.get(event_url, headers = {'User-Agent' : UserAgent().random})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            rows = soup.find_all('tr', class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
            rows = [row['data-link'] for row in rows]
            print(f"Fight links {len(rows)}")
            return rows
        except:
            time.sleep(60)

def get_fighter_ids(session, fight_links):
    fighter_id_data = []
    def get_ids(session, link):
        try:
            response = session.get(link, headers = {'User-Agent' : UserAgent().random})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            fighters = soup.find_all('a', class_='b-link b-fight-details__person-link')
            r_id = fighters[0]['href'][-16:]
            b_id = fighters[1]['href'][-16:]
            r_name = fighters[0].get_text(strip=True)
            b_name = fighters[1].get_text(strip=True)
            with lock:
                fighter_id_data.append({'r_id': r_id, 'r_name': r_name, 'b_id': b_id, 'b_name': b_name})
        except:
            time.sleep(60)

    with ThreadPoolExecutor(max_workers=10) as executor:        
        results = [executor.submit(get_ids, session, link) for link in fight_links]
        for r in results:
            r.result()
            
    return fighter_id_data

def predict_winners(fighter_id_data):
    df = pd.DataFrame(data=fighter_id_data)
    df_swapped = pd.DataFrame(columns=['r_id', 'b_id'])
    pred_table = df.copy()

    df_swapped['r_id'] = df['b_id']
    df_swapped['b_id'] = df['r_id']

    df = df.merge(r_df, on='r_id')
    df = df.merge(b_df, on='b_id')
    df_swapped = df_swapped.merge(r_df, on='r_id')
    df_swapped = df_swapped.merge(b_df, on='b_id')

    df = df.dropna()
    df_swapped = df_swapped.dropna()

    pred = model.predict(df)
    pred_swapped = model.predict(df_swapped)

    red_fighters = df['r_id'].to_list()
    blue_fighters = df['b_id'].to_list()
    red_fighters_swapped = df_swapped['r_id'].to_list()
    blue_fighters_swapped = df_swapped['b_id'].to_list()

    winner_ids = [red_fighters[idx] if int(num) == 1 else blue_fighters[idx] for idx, num in enumerate(pred)]
    winner_ids_swapped = [red_fighters_swapped[idx] if int(num) == 1 else blue_fighters_swapped[idx] for idx, num in enumerate(pred_swapped)]

    final_predictions = [winner_ids[i] for i in range(len(winner_ids)) if winner_ids[i] == winner_ids_swapped[i]]

    prediction_set = set(final_predictions)

    def find_winner(row):
        """
        Checks if the red or blue fighter ID is in the predictions list.
        Returns a pandas Series containing the winner's ID and name.
        """
        if row['r_id'] in prediction_set:
            return pd.Series([row['r_id'], row['r_name']], index=['winner_id', 'winner_name'])
        elif row['b_id'] in prediction_set:
            return pd.Series([row['b_id'], row['b_name']], index=['winner_id', 'winner_name'])
        else:
            return pd.Series([None, None], index=['winner_id', 'winner_name'])

    pred_table[['winner_id', 'winner_name']] = pred_table.apply(find_winner, axis=1)
    return pred_table

@app.route('/predict_upcoming_event', methods=['GET'])
def predict_upcoming_event():
    session = create_session()
    event_url, event_name = get_event_link(session)
    fight_links = get_fight_links(session, event_url)
    fighter_data = get_fighter_ids(session, fight_links)
    result_df = predict_winners(fighter_data)

    result = result_df[['r_id', 'r_name', 'b_id', 'b_name', 'winner_id', 'winner_name']].to_dict(orient='records')
    return jsonify({'event': event_name, 'predictions': result})


@app.route('/fighter-details', methods=['GET'])
def get_fighter_details():
    fighter_id = request.args.get('fighter_id', '').strip()

    if not fighter_id:
        return jsonify({"error": "Missing 'fighter_id' parameter"}), 400

    fighter_detail = df_fighter[df_fighter['id'] == fighter_id]

    if fighter_detail.empty:
        return jsonify({"error": f"No fighter found with id: {fighter_id}"}), 404
    
    return jsonify(fighter_detail.to_dict(orient='records')[0])


@app.route('/')
def home():
    return '<h1>API WORKING!!</h1>'

@app.route('/event-details', methods=['GET'])
def get_event_details_pagen():
    page = int(request.args.get('page', 1))
    per_page = 10
    total_items = len(df_events)
    total_pages = (total_items + per_page - 1) // per_page

    if page < 1 or page > total_pages:
        return jsonify({
            "error": "Page out of range",
            "total_pages": total_pages,
            "requested_page": page
        }), 404

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_df = df_events.iloc[start_idx:end_idx]

    return jsonify({
        "page": page,
        "total_pages": total_pages,
        "results": page_df.to_dict(orient='records')
    })
    
@app.route('/event-details-by-id', methods=['GET'])
def get_event_details_by_id():
    event_id = request.args.get('event_id')

    if not event_id:
        return jsonify({"error": "Missing 'event_id' parameter"}), 400

    event = df_events[df_events['event_id'] == event_id]

    if event.empty:
        return jsonify({"error": "Event not found", "event_id": event_id}), 404

    return jsonify(event.to_dict(orient='records')[0])


@app.route('/search-fighter', methods=['GET'])
def search_fighter():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400
    matches = df_fighter[df_fighter['name'].str.lower().str.contains(query)]
    results = matches[['id', 'name']].head(15).to_dict(orient='records')
    return jsonify({"query": query, "results": results})

@app.route('/predict-single-fight', methods=['GET'])
def predict_single_fight():
    r_id = request.args.get('red_id', '').strip()
    b_id = request.args.get('blue_id', '').strip()

    if not r_id or not b_id:
        return jsonify({"error": "Both 'red_id' and 'blue_id' must be provided"}), 400
    fighter_data = [{
        'r_id': r_id,
        'r_name': df_fighter[df_fighter['id'] == r_id]['name'].values[0] if not df_fighter[df_fighter['id'] == r_id].empty else 'Unknown',
        'b_id': b_id,
        'b_name': df_fighter[df_fighter['id'] == b_id]['name'].values[0] if not df_fighter[df_fighter['id'] == b_id].empty else 'Unknown',
    }]

    try:
        result_df = predict_winners(fighter_data)
    except Exception as e:
        return jsonify({
            "winner": "Cannot conclude",
            "reason": "Model predictions do not agree or missing data"
        }), 500

    winner_row = result_df.iloc[0]

    if pd.isna(winner_row['winner_id']):
        return jsonify({
            "winner": "Inconclusive",
            "reason": "Model predictions do not agree or missing data"
        })

    return jsonify({
        "red_id": r_id,
        "red_name": winner_row['r_name'],
        "blue_id": b_id,
        "blue_name": winner_row['b_name'],
        "winner_id": winner_row['winner_id'],
        "winner_name": winner_row['winner_name']
    })

    
if __name__ == '__main__':
    app.run()