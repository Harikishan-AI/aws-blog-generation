import boto3
import botocore.config
import json

from datetime import datetime
from typing import Optional, List

# CrewAI imports for multi-step orchestration using Bedrock via LiteLLM
try:
    from crewai import Agent, Task, Crew, Process, LLM
except Exception:
    # Defer import errors to runtime use to keep cold-start imports lightweight
    Agent = Task = Crew = Process = LLM = None


def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt=f"""<s>[INST]Human: Write a 200 words blog on the topic {blogtopic}
    Assistant:[/INST]
    """

    body={
        "prompt":prompt,
        "max_gen_len":512,
        "temperature":0.5,
        "top_p":0.9
    }

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}),
        )
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama2-13b-chat-v1")

        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        print(response_data)
        blog_details = response_data['generation']
        return blog_details
    except Exception as e:
        print(f"Error generating the blog:{e}")
        return ""


def blog_generate_using_crewai_content_marketing(
    blogtopic: str,
    brand_name: Optional[str] = None,
    target_audience: Optional[str] = None,
    tone: Optional[str] = None,
    seo_keywords: Optional[List[str]] = None,
    target_word_count: int = 700,
) -> str:
    """Multi-step orchestration using CrewAI: research → outline → write → edit.

    Falls back to direct Bedrock call if CrewAI is unavailable.
    """
    # Lazy import in case CrewAI is not installed in the runtime
    if any(x is None for x in [Agent, Task, Crew, Process, LLM]):
        print("CrewAI not available; falling back to direct Bedrock call.")
        return blog_generate_using_bedrock(blogtopic)

    bedrock_llm = LLM(
        # Uses LiteLLM under the hood to route to AWS Bedrock
        model="bedrock/meta.llama2-13b-chat-v1",
        temperature=0.5,
        max_tokens=2048,
        aws_region_name="us-east-1",
        timeout=300,
    )

    audience_str = target_audience or "a general business audience"
    tone_str = tone or "helpful, expert, and approachable"
    brand_str = brand_name or "our brand"
    keywords_str = ", ".join(seo_keywords) if seo_keywords else ""

    researcher = Agent(
        role="Market Research Analyst",
        goal=(
            "Research the topic, identify up-to-date facts, trends, statistics, pain points, and SEO keywords."
        ),
        backstory=(
            "You are a meticulous B2B/B2C market researcher who summarizes credible insights succinctly."
        ),
        allow_delegation=False,
        llm=bedrock_llm,
    )

    strategist = Agent(
        role="Content Strategist",
        goal="Design a high-converting outline aligned with the audience, brand, and SEO goals.",
        backstory="You structure content to maximize clarity, search intent match, and engagement.",
        allow_delegation=False,
        llm=bedrock_llm,
    )

    writer = Agent(
        role="Senior Copywriter",
        goal="Write persuasive, clear copy that is accurate and on-brand.",
        backstory="You write concise, engaging articles with smooth flow and strong transitions.",
        allow_delegation=False,
        llm=bedrock_llm,
    )

    editor = Agent(
        role="Managing Editor",
        goal=(
            "Edit for accuracy, coherence, brand tone, grammar, and SEO. Ensure factual consistency and polish."
        ),
        backstory="You are uncompromising on quality and clarity; you deliver publication-ready content.",
        allow_delegation=False,
        llm=bedrock_llm,
    )

    research_task = Task(
        description=(
            f"Conduct research for a blog about '{blogtopic}'.\n"
            f"Audience: {audience_str}. Brand: {brand_str}. Desired tone: {tone_str}.\n"
            f"If SEO keywords provided, prioritize them: {keywords_str if keywords_str else 'none provided'}.\n"
            "Deliver: \n"
            "- 6-10 bullet points with key insights, stats (with approximate figures), and pain points\n"
            "- 8-12 SEO keyword ideas (short and long-tail)\n"
            "- 3-5 proposed angles for the article\n"
        ),
        expected_output=(
            "Bullet list of insights, keyword list, and angles suitable for planning the article."
        ),
        agent=researcher,
    )

    outline_task = Task(
        description=(
            "Create a detailed outline using the research. Include: title options, H2/H3 sections,"
            " bullet notes per section, and an SEO snippet plan (title tag + meta description)."
        ),
        expected_output=(
            "A structured outline with 1-2 title options, 5-8 H2s (with optional H3s), and bullet notes."
        ),
        agent=strategist,
        context=[research_task],
    )

    write_task = Task(
        description=(
            f"Write a {target_word_count}-word article based on the outline. Maintain {tone_str} tone for {audience_str}. "
            f"We are {brand_str}. Incorporate the most important keywords naturally and avoid keyword stuffing."
        ),
        expected_output=(
            "A cohesive article with intro, sections per outline, and a conclusion. No outline or notes, only prose."
        ),
        agent=writer,
        context=[research_task, outline_task],
    )

    edit_task = Task(
        description=(
            "Revise the drafted article for clarity, correctness, brand voice, and SEO."
            " Ensure factual consistency with the research. Add a short meta description (<= 160 chars) and a CTA."
            " Return only the final publication-ready article text (followed by the meta description and CTA)."
        ),
        expected_output=(
            "Final polished article text suitable for publishing, then a 'Meta Description:' and 'CTA:' section."
        ),
        agent=editor,
        context=[research_task, outline_task, write_task],
    )

    crew = Crew(
        agents=[researcher, strategist, writer, editor],
        tasks=[research_task, outline_task, write_task, edit_task],
        process=Process.sequential,
    )

    result = crew.kickoff()

    # Try to extract the editor's final output if available
    try:
        final_text = getattr(edit_task, "output", None)
        final_text = getattr(final_text, "raw", None) if final_text is not None else None
        if isinstance(final_text, str) and final_text.strip():
            return final_text.strip()
    except Exception:
        pass

    return str(result).strip()

def save_blog_details_s3(s3_key, s3_bucket, generate_blog):
    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generate_blog)
        print("Content saved to s3")
    except Exception as e:
        print("Error when saving the content to s3", e)



def lambda_handler(event, context):
    # Accept API Gateway input. Expected body JSON with at least 'blog_topic'.
    event = json.loads(event['body'])
    blogtopic = event['blog_topic']

    brand = event.get('brand_name')
    audience = event.get('target_audience')
    tone = event.get('tone')
    seo_keywords = event.get('seo_keywords')  # optional list
    word_count = event.get('target_word_count', 700)

    # Use multi-step CrewAI pipeline; fallback internally to direct Bedrock if needed
    generate_blog = blog_generate_using_crewai_content_marketing(
        blogtopic=blogtopic,
        brand_name=brand,
        target_audience=audience,
        tone=tone,
        seo_keywords=seo_keywords,
        target_word_count=word_count,
    )

    if generate_blog:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f"blog-output/{current_time}.txt"
        s3_bucket = 'aws_bedrock_course1'
        save_blog_details_s3(s3_key, s3_bucket, generate_blog)
    else:
        print("No blog was generated")

    return {
        'statusCode': 200,
        'body': json.dumps('Blog generation is completed')
    }

    




