from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Import logic from src (Assuming Airflow paths are set correctly)
# In production, use DockerOperator or KubernetesPodOperator for isolation
# from src.models.train_churn import ChurnTrainer
# from src.models.train_recsys import RecSysTrainer

def run_churn_training():
    print("Simulating Churn Model Training...")
    # trainer = ChurnTrainer()
    # trainer.train()

def run_recsys_training():
    print("Simulating RecSys Model Training...")
    # trainer = RecSysTrainer()
    # trainer.train()

default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retail_customer_360_pipeline',
    default_args=default_args,
    description='Daily training of Churn and RecSys models',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
) as dag:

    train_churn_task = PythonOperator(
        task_id='train_churn_model',
        python_callable=run_churn_training,
    )

    train_recsys_task = PythonOperator(
        task_id='train_recsys_model',
        python_callable=run_recsys_training,
    )

    # Parallel Execution
    [train_churn_task, train_recsys_task]