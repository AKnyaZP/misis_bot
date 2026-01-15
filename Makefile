.PHONY: help build up down logs restart parser clean

help: ## Показать справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Собрать Docker образы
	docker-compose build

up: ## Запустить все сервисы
	docker-compose up -d

down: ## Остановить все сервисы
	docker-compose down

logs: ## Показать логи всех сервисов
	docker-compose logs -f

logs-bot: ## Показать логи бота
	docker-compose logs -f bot

logs-parser: ## Показать логи парсера
	docker-compose logs -f parser

restart: ## Перезапустить все сервисы
	docker-compose restart

parser: ## Запустить парсер для загрузки данных
	docker-compose run --rm parser

rebuild: ## Пересобрать и перезапустить
	docker-compose down
	docker-compose build
	docker-compose up -d

clean: ## Остановить и удалить все контейнеры и volumes
	docker-compose down -v

status: ## Показать статус сервисов
	docker-compose ps
