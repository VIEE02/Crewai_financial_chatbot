from venv import logger
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
from crewai_tools import DirectoryReadTool
from crewai_tools import NL2SQLTool
import logging
import yaml

@CrewBase
class Financial():
    """Financial crew"""
    def __init__(self):
        # Load agent configs
        try:
            with open('src/financial/config/agents.yaml', 'r') as file:
                self.agent_configs = yaml.safe_load(file)
            logger.info("Agent configs loaded successfully")
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            raise

    @agent
    def chat_input(self) -> Agent:
        return Agent(
            config=self.agents_config['chat_input'],
            tools=[],
        )
    @agent
    def chat_input_agent(self) -> Agent:
        return Agent(
            role=self.agent_configs['chat_input']['role'],
            goal="Analyze user query and extract relevant information like company symbols",
            backstory="I am an expert at understanding financial queries and extracting key information",
            tools=[DirectoryReadTool()],
            verbose=True,
            allow_delegation=True
        )

    @agent
    def query_analysis(self) -> Agent:
        return Agent(
            config=self.agents_config['query_analysis'],
            tools=[],
        )

    @agent
    def data_loader(self) -> Agent:
        return Agent(
            config=self.agents_config['data_loader'],
            tools=[],
        )
    @agent
    def data_loader_agent(self) -> Agent:
        return Agent(
            role="Financial Data Loader",
            goal="Load and prepare financial data for analysis",
            backstory="I specialize in gathering and organizing financial data",
            tools=[CSVSearchTool(), DirectoryReadTool()],
            verbose=True
        )

    @agent
    def historical_financial_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['historical_financial_expert'],
            tools=[],
        )

    @agent
    def api_response(self) -> Agent:
        return Agent(
            config=self.agents_config['api_response'],
            tools=[],
        )
    @agent
    def analyst_agent(self) -> Agent:
        return Agent(
            role="Financial Analyst",
            goal="Analyze financial data and provide insights",
            backstory="I am an expert financial analyst",
            verbose=True
        )

    @task
    def fastapi_endpoint_setup(self) -> Task:
        return Task(
            config=self.tasks_config['fastapi_endpoint_setup'],
            tools=[],
        )

    @task
    def request_parsing_validation(self) -> Task:
        return Task(
            config=self.tasks_config['request_parsing_validation'],
            tools=[],
        )

    @task
    def llm_query_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['llm_query_analysis_task'],
            tools=[],
        )

    @task
    def historical_data_loading_task(self) -> Task:
        return Task(
            config=self.tasks_config['historical_data_loading_task'],
            tools=[CSVSearchTool(), DirectoryReadTool()],
        )

    @task
    def historical_query_processing_task(self) -> Task:
        return Task(
            config=self.tasks_config['historical_query_processing_task'],
            tools=[],
        )

    @task
    def response_packaging_return_task(self) -> Task:
        return Task(
            config=self.tasks_config['response_packaging_return_task'],
            tools=[],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewAutomationChatbotFastapiIntegration crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
