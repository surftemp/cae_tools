
import sqlite3
import os
import datetime
import json

SCHEMA_VERSION = "0.1"

class ModelDatabase:

    def __init__(self,database_path):
        if not os.path.exists(database_path):
            self.conn = sqlite3.connect(database_path)
            curs = self.conn.cursor()
            curs.execute(
                "CREATE TABLE MODEL_SCHEMA(version STRING)")
            curs.execute("INSERT INTO MODEL_SCHEMA VALUES (?)",(SCHEMA_VERSION,))
            curs.execute(
                "CREATE TABLE MODEL_TRAINING(timestamp DATE, model_id STRING, model_type STRING, target_variable STRING, input_variables STRING, model_description STRING, model_path STRING, train_path STRING, train_loss FLOAT, test_path STRING, test_loss FLOAT, hyperparameters STRING, spec STRING)")
            curs.execute(
                "CREATE TABLE MODEL_EVALUATIONS(timestamp DATE, model_id STRING, train_path STRING, test_path STRING, metrics STRING)")
            self.conn.commit()
        else:
            self.conn = sqlite3.connect(database_path)

    def add_training_result(self, model_id, model_type, target_variable, input_variables, description, model_path, train_path, train_loss, test_path, test_loss, hyperparameters, spec):
        curs = self.conn.cursor()
        curs.execute("INSERT INTO MODEL_TRAINING VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            datetime.datetime.now(), model_id, model_type, target_variable, json.dumps(input_variables), description, model_path, train_path, train_loss, test_path, test_loss, json.dumps(hyperparameters),
            json.dumps(spec)))
        self.conn.commit()

    def add_evaluation_result(self, model_id, train_path, test_path, metrics):
        curs = self.conn.cursor()
        dt_now = datetime.datetime.now()
        curs.execute("INSERT INTO MODEL_EVALUATIONS VALUES(?,?,?,?,?)", (
            dt_now, model_id, train_path, test_path, json.dumps(metrics)))
        self.conn.commit()

    def dump(self):
        curs = self.conn.cursor()
        print("MODEL_SCHEMA")
        rs = curs.execute("SELECT * FROM MODEL_SCHEMA").fetchall()
        for row in rs:
            print(json.dumps(row))
        print("\nMODEL_TRAINING")
        rs = curs.execute("SELECT * FROM MODEL_TRAINING").fetchall()
        for row in rs:
            print(json.dumps(row))
        print("\nMODEL_EVALUTATIONS")
        rs = curs.execute("SELECT * FROM MODEL_EVALUATIONS").fetchall()
        for row in rs:
            print(json.dumps(row))



