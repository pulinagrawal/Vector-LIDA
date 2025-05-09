@echo off
echo Starting batch test script for multiple agent configurations
echo ===========================================================

:: Set common paths
set VIDEO_PATH=c:\Users\Nathan\CVResearch\Vector-LIDA\film_agent\test_data\videos\text1.mp4
set LABELS_PATH=c:\Users\Nathan\CVResearch\Vector-LIDA\film_agent\test_data\labels\test1.json
set BATCH_NAME=batch4
set RESULTS_DIR=film_agent\test_data\results\%BATCH_NAME%

:: Create the batch directory if it doesn't exist
if not exist %RESULTS_DIR% mkdir %RESULTS_DIR%

echo Results will be saved to: %RESULTS_DIR%
echo Video path: %VIDEO_PATH%
echo Labels path: %LABELS_PATH%
echo.

:: Run static agent test (baseline)
echo Running static agent test...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent static --output %RESULTS_DIR%\static_results.json
echo.

:: --------------- Adaptive Agent Tests ---------------

:: Test 1: Adaptive agent with no initial embeddings and confidence threshold 0.1
echo Running adaptive agent test 1: No initial embeddings, confidence=0.1...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.1 --output %RESULTS_DIR%\adaptive_no_ref_conf01.json
echo.

:: Test 2: Adaptive agent with no initial embeddings and confidence threshold 0.3
echo Running adaptive agent test 2: No initial embeddings, confidence=0.3...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.3 --output %RESULTS_DIR%\adaptive_no_ref_conf03.json
echo.

:: Test 3: Adaptive agent with no initial embeddings and confidence threshold 0.5
echo Running adaptive agent test 3: No initial embeddings, confidence=0.5...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.5 --output %RESULTS_DIR%\adaptive_no_ref_conf05.json
echo.

:: Test 4: Adaptive agent with no initial embeddings and confidence threshold 0.7
echo Running adaptive agent test 4: No initial embeddings, confidence=0.7...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.7 --output %RESULTS_DIR%\adaptive_no_ref_conf07.json
echo.

:: Test 5: Adaptive agent with no initial embeddings and confidence threshold 0.9
echo Running adaptive agent test 5: No initial embeddings, confidence=0.9...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.9 --output %RESULTS_DIR%\adaptive_no_ref_conf09.json
echo.

:: Tests with initial embeddings
:: Test 6: Adaptive agent WITH initial embeddings and confidence threshold 0.1
echo Running adaptive agent test 6: With initial embeddings, confidence=0.1...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.1 --use_initial_embeddings --output %RESULTS_DIR%\adaptive_with_ref_conf01.json
echo.

:: Test 7: Adaptive agent WITH initial embeddings and confidence threshold 0.3
echo Running adaptive agent test 7: With initial embeddings, confidence=0.3...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.3 --use_initial_embeddings --output %RESULTS_DIR%\adaptive_with_ref_conf03.json
echo.

:: Test 8: Adaptive agent WITH initial embeddings and confidence threshold 0.5
echo Running adaptive agent test 8: With initial embeddings, confidence=0.5...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.5 --use_initial_embeddings --output %RESULTS_DIR%\adaptive_with_ref_conf05.json
echo.

:: Test 9: Adaptive agent WITH initial embeddings and confidence threshold 0.7
echo Running adaptive agent test 9: With initial embeddings, confidence=0.7...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.7 --use_initial_embeddings --output %RESULTS_DIR%\adaptive_with_ref_conf07.json
echo.

:: Test 10: Adaptive agent WITH initial embeddings and confidence threshold 0.9
echo Running adaptive agent test 10: With initial embeddings, confidence=0.9...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --confidence_threshold 0.9 --use_initial_embeddings --output %RESULTS_DIR%\adaptive_with_ref_conf09.json
echo.

:: --------------- EMA Agent Tests ---------------

:: Test 4: EMA agent with no initial embeddings and varying alpha and confidence thresholds
echo Running EMA agent tests with combinations of alpha and confidence thresholds...

:: Loop through alpha values (without initial embeddings)
for %%A in (0.1 0.5 0.9) do (
    :: Loop through confidence threshold values
    for %%C in (0.1 0.5 0.9) do (
        echo Running EMA agent test: alpha=%%A, confidence=%%C...
        python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent ema --ema_alpha %%A --confidence_threshold %%C --output %RESULTS_DIR%\ema_alpha%%A_conf%%C.json
        echo.
    )
)

:: Test 5: EMA agent WITH initial embeddings and varying alpha and confidence thresholds
echo Running EMA agent tests with initial embeddings, combinations of alpha and confidence thresholds...

:: Loop through alpha values (with initial embeddings)  
for %%A in (0.1 0.5 0.9) do (
    :: Loop through confidence threshold values
    for %%C in (0.1 0.5 0.9) do (
        echo Running EMA agent test with initial embeddings: alpha=%%A, confidence=%%C...
        python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent ema --ema_alpha %%A --confidence_threshold %%C --use_initial_embeddings --output %RESULTS_DIR%\ema_with_ref_alpha%%A_conf%%C.json
        echo.
    )
)

:: --------------- Self Learning Agent Tests ---------------

:: Test 7: Self Learning agent with default settings
echo Running Self Learning agent test 1: Default settings...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent selflearning --output %RESULTS_DIR%\selflearning_default.json
echo.

:: Test 8: Self Learning agent with custom motion threshold
echo Running Self Learning agent test 2: Custom motion threshold...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent selflearning --motion_threshold 0.07 --bootstrapping_frames 30 --output %RESULTS_DIR%\selflearning_motion007_boot30.json
echo.

:: --------------- Test with custom bootstrapping settings ---------------

:: Test 9: Adaptive agent with custom bootstrapping
echo Running adaptive agent test 4: Custom bootstrapping settings...
python film_agent/test_agent_accuracy.py --video %VIDEO_PATH% --labels %LABELS_PATH% --agent adaptive --bootstrapping_frames 40 --motion_threshold 0.08 --confidence_threshold 0.1 --output %RESULTS_DIR%\adaptive_custom_bootstrap.json
echo.

:: --------------- Complete ---------------

echo.
echo All tests completed!
echo Results saved in: %RESULTS_DIR%
echo.

:: Wait for user input before closing
pause