@echo off
echo Creating Model Performance Visualizations...
echo ==========================================

cd /d "%~dp0"

echo Running final model performance visualizer...
python scripts/visualization/final_model_performance_visualizer.py

echo.
echo Visualizations created successfully!
echo Check the outputs/visualizations directory for generated images.
echo.
echo Generated files:
echo - final_performance_table.png (Performance comparison table)
echo - final_learning_curves.png (Learning curves for top models)
echo - final_confusion_matrices.png (Confusion matrices for validation set)
echo.
pause
