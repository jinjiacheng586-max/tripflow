# TripFlow

A Streamlit-based NSW rail journey planning prototype focused on transfer reliability.

## Project Overview
TripFlow is a journey planning prototype for NSW public transport.  
It helps users choose not just a possible route, but a more reliable one.

The current prototype focuses on rail-based journeys and supports:
- direct route search
- one-transfer route search
- transfer buffer analysis
- reliability scoring
- best route recommendation

## Features
- Station-level search
- Departure time input
- Direct and one-transfer journey search
- Transfer buffer classification:
  - Safe
  - Risky
  - Very Tight
- Reliability-based recommendation
- Streamlit web interface

## Example Demo Cases
- Direct journey: Tallawong Station -> Epping Station
- One-transfer journey: Ashfield Station -> Mascot Station

## Project Structure
- `app.py` — Streamlit frontend
- `requirements.txt` — project dependencies
- `src/load_data.py` — GTFS data loading
- `src/preprocess.py` — preprocessing logic
- `src/route_search.py` — direct route search
- `src/transfer_search.py` — one-transfer search
- `src/scoring.py` — reliability scoring
- `src/recommendation.py` — route ranking and recommendation
- `screenshots/` — demo images

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
