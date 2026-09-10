from techniques import (
    orchestrator_parallel,
    orchestrator_sequential,
    sequential_pipeline,
    single_agent_react,
    supervisor_verification,
)

ARCHITECTURES = {
    "single_agent_react": single_agent_react,
    "sequential_pipeline": sequential_pipeline,
    "orchestrator_sequential": orchestrator_sequential,
    "orchestrator_parallel": orchestrator_parallel,
    "supervisor_verification": supervisor_verification,
}
