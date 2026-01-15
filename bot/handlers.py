
from huggingface_hub import InferenceClient
from requests import get_hf_response

from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram import Router, F
from aiogram.filters import CommandStart, StateFilter
from aiogram.types import Message


router = Router()

# можно добавлять этапы составления или валидации вопроса
class RequestCompositing(StatesGroup):
    question = State()



@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.set_state(RequestCompositing.question)
    await message.answer("Привет! Задавай свой вопрос!)")

@router.message(F.text, RequestCompositing.question)
async def generate(message: Message, state: FSMContext, hf_client: InferenceClient):
    await state.set_state('generating')
    await message.answer("Начинаю генерацию ответа!")

    prompt = message.text

    try:
        response_text = await get_hf_response(prompt, hf_client)
    except Exception as e:
        await message.answer(f"Ошибка при запросе к серверу: {e}")
    else:
        await message.answer(response_text)
    finally:
        await state.clear()

@router.message(StateFilter('generating'))
async def wait_responce(message : Message) -> None:
    await message.answer("Ожидайте завершения генерации ответа на ваш вопрос")