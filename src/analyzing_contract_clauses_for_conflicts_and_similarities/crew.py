from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from typing import Optional, Dict, Any

# from crewai_tools import QdrantVectorSearchTool
import os
from dotenv import load_dotenv
from analyzing_contract_clauses_for_conflicts_and_similarities.tools.qdrant_vector_search_tool import (
    QdrantVectorSearchTool
)
from analyzing_contract_clauses_for_conflicts_and_similarities.tools.pre_process_docs import (
    pre_process_docs
)

load_dotenv()


@CrewBase
class AnalyzingContractClausesForConflictsAndSimilaritiesCrew:
    """AnalyzingContractClausesForConflictsAndSimilarities crew"""

    vector_search_tool = QdrantVectorSearchTool(
        collection_name="contracts_business_5",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )

    @before_kickoff
    def pre_process_docs(self, inputs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run preprocessing before the crew starts."""
        if inputs is None:
            return None

        # Get force_reindex from inputs if provided
        force_reindex = inputs.get('force_reindex', False)
        
        # Run preprocessing - it will automatically handle new files
        pre_process_docs(force_reindex=force_reindex)
        return inputs

    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["source_citer_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],
            tools=[self.vector_search_tool],
        )

    @task
    def retrieve_contracts_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_contracts_task"],
        )

    @task
    def source_citer_task(self) -> Task:
        return Task(
            config=self.tasks_config["source_citer_task"],
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_report_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalyzingContractClausesForConflictsAndSimilarities crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
