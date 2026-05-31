# Agentic AI Systems

## What is an AI agent?

An AI agent is a system that uses a large language model not just to generate text, but
to decide and take actions in pursuit of a goal. Instead of producing a single response,
an agent runs in a loop: it observes the current situation, reasons about what to do
next, takes an action (such as calling a tool or searching a database), observes the
result, and repeats until the goal is achieved. This loop of reasoning and acting is
what makes a system "agentic."

## Core components of an agent

Most agentic systems share four components. First, a model that does the reasoning.
Second, a set of tools the agent can call, such as a search function, a calculator, or a
database query. Third, a memory that lets the agent keep track of what it has already
done and learned during the task. Fourth, an orchestration or control loop that decides
when to act, when to reflect, and when to stop. Together these let the agent break a
complex task into steps and adapt as it goes.

## Key characteristics

Agentic systems are characterized by autonomy (they make decisions without a human in
every step), the ability to plan (breaking a goal into sub-steps), tool use (interacting
with external systems), and reflection (evaluating their own intermediate results and
correcting course). A defining trait is the feedback loop: the agent can look at the
outcome of an action and decide to try something different, rather than blindly
following a fixed script.

## Planning and reflection

Planning is the agent's ability to decide a sequence of steps before or during
execution. Reflection, sometimes called self-correction, is the agent's ability to
critique its own output and improve it. For example, after retrieving information, an
agent might grade whether that information is actually relevant, and if not, reformulate
its request and try again. This self-correcting behavior is the key advantage of agentic
approaches over rigid, single-pass pipelines.

## Agentic RAG

Agentic RAG applies these agentic principles to retrieval-augmented generation. Rather
than retrieving once and answering, an agentic RAG system can decide whether retrieval
is needed at all, rewrite the query to improve retrieval, grade the relevance of
retrieved chunks, retrieve again if the first attempt was poor, and verify that the
final answer is grounded in the sources before returning it. This turns RAG from a fixed
pipeline into an adaptive, self-correcting process.
