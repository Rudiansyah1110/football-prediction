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

## Example Result
$ python run_prediction.py
Running prediction pipeline...

Pipeline progress: 100%|##########| 5/5 [00:02<00:00,  2.48it/s]

=== CONFUSION MATRIX ===
[[ 794  766  836]
 [ 675 1821  454]
 [ 875  615 2370]]

=== CLASSIFICATION REPORT ===
              precision    recall  f1-score   support

           D       0.34      0.33      0.34      2396
           L       0.57      0.62      0.59      2950
           W       0.65      0.61      0.63      3860

    accuracy                           0.54      9206
   macro avg       0.52      0.52      0.52      9206
weighted avg       0.54      0.54      0.54      9206


=== PROBABILITIES ===
   prob_under_3_5  prob_over_3_5
0        0.736313       0.263687
1        0.768597       0.231403
2        0.435326       0.564674
3        0.766227       0.233773
4        0.801883       0.198117
5        0.719304       0.280696
6        0.792007       0.207993
7        0.912619       0.087381

=== FUTURE MATCH PREDICTION ===
Predicted Team: Liverpool VS Tottenham
Predicted Result: D
Probabilities:
  D: 0.42
  L: 0.25
  W: 0.33

=== FEATURE WEIGHTS ===
                           class_D   class_L   class_W
is_home                   0.016536 -0.114337  0.097801
team_avg_goals_for_5     -0.001647 -0.020269  0.021916
team_avg_goals_against_5 -0.025301  0.034813 -0.009512
opp_avg_goals_for_5      -0.014814  0.012735  0.002080
opp_avg_goals_against_5   0.005025 -0.007770  0.002745
team_shots                0.052881  0.165456 -0.218337
team_sot                 -0.062448 -0.617251  0.679699
opp_shots                 0.046910 -0.214978  0.168068
opp_sot                  -0.091338  0.676707 -0.585369

## How to Run
```bash
pip install -r requirements.txt
python run_prediction.py

