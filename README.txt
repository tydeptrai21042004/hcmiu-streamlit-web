This ZIP contains the four CALF source files requested:

CALF/run.py
CALF/models/CALF.py
CALF/models/GPT2_arch.py
CALF/exp/exp_long_term_forecasting.py

Notes:
- run.py is from the public Hank0626/CALF repository.
- CALF.py includes the missing get_peft_model import fix.
- GPT2_arch.py is simplified to use the current HuggingFace GPT2Model forward safely.
- exp_long_term_forecasting.py includes safer eval/train handling and saves input.npy during testing.

Place the CALF folder beside your Streamlit app.py, so the path is:

hcmiu-streamlit-web/
  app.py
  CALF/
    run.py
    models/CALF.py
    models/GPT2_arch.py
    exp/exp_long_term_forecasting.py
