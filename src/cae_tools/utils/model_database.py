
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
        print(model_id,train_path,test_path,metrics)
        curs.execute("INSERT INTO MODEL_EVALUATIONS VALUES(?,?,?,?,?)", (
            dt_now, model_id, train_path, test_path, json.dumps(metrics)))
        self.conn.commit()

    def collect_results(self, curs):
        rows = []
        column_names = [column[0] for column in curs.description]
        for row in curs.fetchall():
            rows.append({v1: v2 for (v1, v2) in zip(column_names, row)})
        return rows

    def expand_training_row(self, row):
        input_vars = json.loads(row["input_variables"])
        return {
            "model_id": row["model_id"],
            "model_type": row["model_type"],
            "input_variables": ", ".join(input_vars),
            "test_loss": "%0.2f"%row["test_loss"],
            "train_loss": "%0.2f"%row["train_loss"]
        }

    def expand_evaluation_row(self, row):
        metrics = json.loads(row["metrics"])
        return {
            "model_id": row["model_id"],
            "test_mse": "%0.2f"%metrics["test"]["mse"],
            "train_mse": "%0.2f"%metrics["train"]["mse"],
            "test_mae": "%0.2f" % metrics["test"]["mae"],
            "train_mae": "%0.2f" % metrics["train"]["mae"]
        }

    def dump_row(self, training_row, evaluation_row=None):
        model_id = training_row["model_id"]
        model_type = training_row['model_type']
        test_loss = training_row['test_loss']
        train_loss = training_row['train_loss']
        input_variables = training_row["input_variables"]
        test_mse = evaluation_row["test_mse"] if evaluation_row is not None else ""
        train_mse = evaluation_row["train_mse"] if evaluation_row is not None else ""
        test_mae = evaluation_row["test_mae"] if evaluation_row is not None else ""
        train_mae = evaluation_row["train_mae"] if evaluation_row is not None else ""
        print(f"| {model_id:36s} | {model_type:9s} | {test_loss:10s} | {train_loss:10s} | {test_mse:10s} | {train_mse:10s} | {test_mae:10s} | {train_mae:10s} | {input_variables}")

    def dump_schema(self):
        curs = self.conn.cursor()
        print("MODEL_SCHEMA")
        rs = curs.execute("SELECT * FROM MODEL_SCHEMA").fetchall()
        for row in rs:
            print(json.dumps(row))
        print()

    def dump_item(self, item, field_names):
        maxlen = 0
        for key in item:
            display_key = field_names.get(key, key)
            maxlen = max(maxlen,len(display_key))

        for key in item:
            display_key = field_names.get(key,key)
            value = item[key]
            padded_key = " "*(maxlen-len(display_key)) + display_key
            if isinstance(value,str) and value.startswith("{"):
                value = json.loads(value)
                lines = json.dumps(value,indent=4).split("\n")
            else:
                lines = str(value).split("\n")
            print(padded_key + ": " + lines[0])
            for line in lines[1:]:
                print(maxlen*" "+"  "+line)


    def dump(self):
        curs = self.conn.cursor()
        rows = self.collect_results(curs.execute("SELECT * FROM MODEL_TRAINING ORDER BY test_loss ASC"))
        self.dump_row({
            "model_id": "ModelID",
            "model_type": "ModelType",
            "test_loss": "Test Loss",
            "train_loss": "Train Loss",
            "input_variables": "Inputs"
        },{
            "test_mse": "Test MSE",
            "train_mse": "Train MSE",
            "test_mae": "Test MAE",
            "train_mae": "Train MAE"
        })
        for row in rows:
            row = self.expand_training_row(row)
            eval_rows = self.collect_results(curs.execute("SELECT * FROM MODEL_EVALUATIONS WHERE model_id=?",[row["model_id"]]))
            if len(eval_rows) == 0:
                self.dump_row(row, None)
            else:
                for idx in range(len(eval_rows)):
                    eval_row = self.expand_evaluation_row(eval_rows[idx])
                    if idx == 0:
                        self.dump_row(row,eval_row)
                    else:
                        self.dump_row({
                            "model_id": "ModelID",
                            "model_type": "ModelType"
                        },eval_row)

        print()

    def dump_model(self, model_id):
        curs = self.conn.cursor()
        print("\n\nModel:")
        rows = self.collect_results(curs.execute("SELECT * FROM MODEL_TRAINING WHERE model_id=?",[model_id]))
        if len(rows) == 0:
            print("Model not found")
            return
        for row in rows:
            self.dump_item(row,{"model_id":"Model ID"})

        print("\n\nModel Evaluations:")
        rows = self.collect_results(curs.execute("SELECT * FROM MODEL_EVALUATIONS WHERE model_id=?",[model_id]))
        if rows:
            for row in rows:
                self.dump_item(row, {"model_id": "Model ID"})
        else:
            print("No evaluations found")


