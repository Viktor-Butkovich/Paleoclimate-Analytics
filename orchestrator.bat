if exist Data\Temp12k-sqlite.db (
    del Data\Temp12k-sqlite.db
)

python3 Temp12k_ETL.py
python3 Temp12k_view_creation.py
Rscript Temp12k_visualization.r
Rscript Temp12k_classical_regression.r