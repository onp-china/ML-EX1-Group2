@echo off
chcp 65001 >nul
echo ========================================
echo 第五阶段高级集成方法快速启动
echo ========================================
echo.

if "%1"=="--all" (
    echo 运行完整对比测试...
    python run_comparison.py --all
) else if "%1"=="--results" (
    echo 显示测试结果...
    python run_comparison.py --results
) else if "%1"=="--method" (
    if "%2"=="mc_dropout" (
        echo 运行MC Dropout测试...
        python run_comparison.py --method mc_dropout
    ) else if "%2"=="two_level_stacking" (
        echo 运行两层Stacking测试...
        python run_comparison.py --method two_level_stacking
    ) else if "%2"=="dynamic_ensemble" (
        echo 运行动态集成测试...
        python run_comparison.py --method dynamic_ensemble
    ) else (
        echo 未知的方法: %2
        echo 可用方法: mc_dropout, two_level_stacking, dynamic_ensemble
    )
) else (
    echo 使用方法:
    echo   run_comparison.bat --all                   运行完整对比测试
    echo   run_comparison.bat --method ^<method^>     运行特定方法测试
    echo   run_comparison.bat --results               显示测试结果
    echo.
    echo 方法选项:
    echo   mc_dropout              MC Dropout + 动态权重
    echo   two_level_stacking      两层Stacking + 动态权重
    echo   dynamic_ensemble        动态集成
    echo.
    echo 示例:
    echo   run_comparison.bat --all
    echo   run_comparison.bat --method mc_dropout
    echo   run_comparison.bat --results
)

pause
