# Football Match Prediction (Machine Learning)

End-to-end machine learning project to predict football match results
(W / D / L) and Over/Under 3.5 goals using historical match data.

## Project Scope
- Data cleansing & standardization
- Rolling features (last 5 matches)
- Match result classification
- Over / Under 3.5 goals prediction
- Future match prediction

## Features Used
- Home / Away indicator
- Average goals scored (last 5 matches)
- Average goals conceded (last 5 matches)
- Shots & shots on target statistics

## Rough summary of column classification
🔵 Identitas:
match_date
opponent
is_home

🟢 Observasi match:
team_goals
opp_goals
team_shots
team_sot
opp_shots
opp_sot

🟡 Rolling features (form):
team_avg_goals_for_5
team_avg_goals_against_5
opp_avg_goals_for_5
opp_avg_goals_against_5
result

🔴 Target:
over_3_5

## Structure
football-prediction/
│
├─ dataset/
│   ├─ dataset_master_football.xlsx   ✅
│   ├─ 2025_premier_liga.xlsx
│   ├─ 2025_la_liga.xlsx
│
├─ run_prediction.py        	👈 ORCHESTRATOR ONLY
├─ model_result.py          	👈 Task A
├─ model_over_under.py      	👈 Task B
├─ cleansing_dataset.py       👈 To clean data 
├─ cleansing_matches.py       👈 To clean data from before
├─ merge_data.py            	👈 Merge data contents from 2 files into 1 master data
└─ utils.py

## How to Run
```bash
pip install -r requirements.txt
python run_prediction.py
