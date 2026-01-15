
from bot.requests import get_hf_response

from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message


router = Router()

# можно добавлять этапы составления или валидации вопроса
class RequestCompositing(StatesGroup):
    question = State()
    generating = State()



@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.set_state(RequestCompositing.question)
    await message.answer("Привет! Задавай свой вопрос!)")

@router.message(F.text, RequestCompositing.question)
async def generate(message: Message, state: FSMContext, hf_service):
    await state.set_state(RequestCompositing.generating)
    await message.answer("Начинаю генерацию ответа!")

    prompt = message.text

    try:
        response_text = await get_hf_response(prompt, hf_service)
    except Exception as e:
        await message.answer(f"Ошибка при запросе к серверу: {e}")
    else:
        await message.answer(response_text)
    finally:
        await state.clear()

@router.message(RequestCompositing.generating)
async def wait_responce(message : Message) -> None:
    await message.answer("Ожидайте завершения генерации ответа на ваш вопрос")