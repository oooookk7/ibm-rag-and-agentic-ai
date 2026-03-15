import os
import yaml
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.tools import (
    ExtractIngredientsTool, 
    FilterIngredientsTool, 
    DietaryFilterTool,
    NutrientAnalysisTool
)

def _build_hf_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        return None
    timeout = float(os.getenv("HUGGINGFACE_HTTP_TIMEOUT", "240"))
    max_tokens = int(os.getenv("HUGGINGFACE_MAX_TOKENS", "1400"))
    return LLM(
        model=os.getenv(
            "HUGGINGFACE_MODEL", "huggingface/Qwen/Qwen3.5-397B-A17B:novita"
        ),
        api_key=token,
        base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        temperature=float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.0")),
        timeout=timeout,
        max_tokens=max_tokens,
    )


HF_LLM = _build_hf_llm()


def _agent_llm_kwargs():
    return {"llm": HF_LLM} if HF_LLM is not None else {}

# Get the absolute path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

@CrewBase
class BaseNourishBotCrew:
    agents_config_path = os.path.join(CONFIG_DIR, 'agents.yaml')
    tasks_config_path = os.path.join(CONFIG_DIR, 'tasks.yaml')
    
    def __init__(self, image_data, dietary_restrictions: str = None):
        self.image_data = image_data
        self.dietary_restrictions = dietary_restrictions

        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)

    @agent
    def ingredient_detection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_detection_agent'],
            tools=[
                ExtractIngredientsTool.extract_ingredient, 
                FilterIngredientsTool.filter_ingredients
            ],
            allow_delegation=False,
            max_iter=5,
            verbose=True,
            **_agent_llm_kwargs()
        )

    @agent
    def dietary_filtering_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['dietary_filtering_agent'],
            tools=[DietaryFilterTool.filter_based_on_restrictions],
            allow_delegation=True,
            max_iter=6,
            verbose=True,
            **_agent_llm_kwargs()
        )

    @agent
    def nutrient_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrient_analysis_agent'],
            tools=[NutrientAnalysisTool.analyze_image],
            allow_delegation=False,
            max_iter=4,
            verbose=True,
            **_agent_llm_kwargs()
        )

    @agent
    def recipe_suggestion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['recipe_suggestion_agent'],
            allow_delegation=False,
            verbose=True,
            **_agent_llm_kwargs()
        )

    @task
    def ingredient_detection_task(self) -> Task:
        task_config = self.tasks_config['ingredient_detection_task']

        return Task(
            description=task_config['description'],
            agent=self.ingredient_detection_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def dietary_filtering_task(self) -> Task:
        task_config = self.tasks_config['dietary_filtering_task']

        return Task(
            description=task_config['description'],
            agent=self.dietary_filtering_agent(),
            depends_on=['ingredient_detection_task'],
            input_data=lambda outputs: {
                'ingredients': outputs['ingredient_detection_task'],
                'dietary_restrictions': self.dietary_restrictions
            },
            expected_output=task_config['expected_output']
        )

    @task
    def nutrient_analysis_task(self) -> Task:
        task_config = self.tasks_config['nutrient_analysis_task']

        return Task(
            description=task_config['description'],
            agent=self.nutrient_analysis_agent(),
            expected_output=task_config['expected_output']
        )

    @task
    def recipe_suggestion_task(self) -> Task:
        task_config = self.tasks_config['recipe_suggestion_task']

        return Task(
            description=task_config['description'],
            agent=self.recipe_suggestion_agent(),
            depends_on=['dietary_filtering_task'],
            input_data=lambda outputs: {
                'filtered_ingredients': outputs['dietary_filtering_task']
            },
            expected_output=task_config['expected_output']
        )


@CrewBase
class NourishBotRecipeCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.ingredient_detection_task(),
            self.dietary_filtering_task(),
            self.recipe_suggestion_task()
        ]

        agents = [
            self.ingredient_detection_agent(),
            self.dietary_filtering_agent(),
            self.recipe_suggestion_agent()
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )


@CrewBase
class NourishBotAnalysisCrew(BaseNourishBotCrew):

    @crew
    def crew(self) -> Crew:
        tasks = [
            self.nutrient_analysis_task(),
        ]

        agents = [
            self.nutrient_analysis_agent(),
        ]

        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
