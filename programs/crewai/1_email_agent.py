from dotenv import load_dotenv
load_dotenv()

from crewai import LLM

llm = LLM(
    model="gemini/gemini-3.1-flash-lite-preview",
    temperature=0.1
)
response = llm.call("Who invented transcendental meditation.")
print(response)

from crewai import Agent, Task, Crew

email_assistant = Agent(
    role="Email Assistant Agent",
    goal="Improve emails and make them sound professional and clear",
    backstory="A highly experienced communication expert skilled in professional email writing",
    verbose=True,
    llm=llm
)

original_email = """
hey team, just wanted to tell u that the demo is kind of ready, but there's still stuff left.
Maybe we can show what we have and say rest is WIP.
Let me know what u think. thanks
"""

email_task = Task(
    description=f"""Take the following rough email and rewrite it into a professional and polished version.
    Expand abbreviations:
    '''{original_email}'''""",
    agent=email_assistant,
    expected_output="A professional written email with proper formatting and content.",
)

crew = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True
)

result = crew.kickoff()
print(result)