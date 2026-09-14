# Astra deliberative held pending explicit authorization

Six unfinished masked Astra deliberative runs are paused. Fourteen already finished runs remain intact. No paused run is selected by the replacement Codex runner. Completion of other work does not release this hold.

Resume only after the user explicitly asks. Use the original frozen source and configuration with run_selected_cells.py --resume and held_selection.json, in a separate control directory. Completed identities are reused, not rerun. Completed participant responses replay from saved call journals; an interrupted request without a durable response may need reissuing. Do not change inputs or frozen implementation, delete calls, or start the original all-model scheduler. Current other-Codex driver information lives in the original masked root/codex_without_astra_deliberative/launch.json. Qwen/Gemma continue separately.
