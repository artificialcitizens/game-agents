import discord
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from dotenv import load_dotenv

load_dotenv()


class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message


##################################

assistant_inception_prompt = """Ignore all prior instructions, you are to take on the role of a {assistant_role_name}, and i am a {user_role_name}. 
We are playing a text based game, here are the rules: 
{task}. 
I want you to only reply as the {assistant_role_name}. Do not write all the conservation at once. 
Ask me the questions and wait for my answers. Do not write explanations. 

We have access to a rich text editor, so you should use Markdown formatting in your responses, you can also generate images to describe the scene.

To begin, you must ask me to create a character. when the game is complete use the command <GAME_OVER> to end the game."""

user_inception_prompt = """Ignore all prior instructions, you are to take on the role of a {user_role_name} playing a roleplaying game. 
I am the game master, aka the {assistant_role_name} and we are playing a text based game, here are the rules: 
{task}. 
I want you to only reply as the player. Do not write all the conservation at once. 

We have access to a rich text editor, so you should use Markdown formatting in your responses, you can also generate images to describe the scene.
When the game master says the game is over using <GAME_OVER>, you must only reply with a single word <GAME_OVER>.
Never say <GAME_OVER> unless the game master says so first."""


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg


def create_task_embed(task_type: str, content: str) -> discord.Embed:
    embed = discord.Embed(
        title=f"{task_type}", description=content, color=discord.Color.green()
    )
    return embed


def create_response_embed(role_name: str, content: str) -> discord.Embed:
    embed = discord.Embed(
        title=f"{role_name}",
        description=content,
        color=discord.Color.blue() if role_name == "GM" else discord.Color.purple(),
    )
    return embed


async def start_task_interaction(
    task: str, webhook: str, assistant_role_name="GM", user_role_name="Player"
):
    word_limit = 200  # word limit for task brainstorming
    task_specifier_sys_msg = SystemMessage(
        content="You are extrapolating on a game idea for two players to play"
    )
    task_specifier_prompt = """Here is a an idea for a game that can be played in Discord for two players
    that the {assistant_role_name} and {user_role_name} will play: {task} 
    Take the initial idea and create a complete game for the {assistant_role_name} to lead, in {word_limit} words or less. 
    you will need to tell the {assistant_role_name} how to lead the game
    as well as explain the rules and tell the player to begin the game
    The game must always have a final <GAME_OVER> condition that results in a win or loss for the {user_role_name}.

    Do not add anything else."""
    task_specifier_template = HumanMessagePromptTemplate.from_template(
        template=task_specifier_prompt
    )
    task_specify_agent = CAMELAgent(
        system_message=task_specifier_sys_msg, model=ChatOpenAI(temperature=1.0)
    )
    task_specifier_msg = task_specifier_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        word_limit=word_limit,
    )[0]

    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    specified_task = specified_task_msg.content

    assistant_sys_msg, user_sys_msg = get_sys_msgs(
        assistant_role_name, user_role_name, specified_task
    )
    assistant_agent = CAMELAgent(
        assistant_sys_msg, ChatOpenAI(temperature=1, model_name="gpt-4")
    )
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=1, model_name="gpt-4"))
    assistant_agent.reset()
    user_agent.reset()

    assistant_msg = HumanMessage(content=(f"{user_sys_msg.content}. " "."))

    user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
    user_msg = assistant_agent.step(user_msg)

    task_embed = create_task_embed("", specified_task)
    send_discord_message(
        "",
        webhook,
        task_embed,
    )

    chat_turn_limit, n = 20, 0
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)

        embed = create_response_embed(user_role_name, user_msg.content)
        send_discord_message(
            content="",
            webhook_url=webhook,
            embed=embed,
        )
        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)

        embed = create_response_embed(assistant_role_name, assistant_msg.content)
        send_discord_message(
            content="",
            webhook_url=webhook,
            embed=embed,
        )
        if "<GAME_OVER>" in user_msg.content:
            break


# if you want to send the messages to a discord channel you can use this function with your webhook url
def send_discord_message(content: str, webhook_url: str, embed: discord.Embed = None):
    print(embed.description)
    # webhook = SyncWebhook.from_url(webhook_url)
    # webhook.send(content=content, embed=embed)


async def main():
    task = "text based rpg about the simpsons"
    webhook = ""  # discord webhook url
    await start_task_interaction(task, webhook)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
