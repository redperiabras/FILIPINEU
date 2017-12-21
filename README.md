# FILIPINEU: Bidirectional Filipino - English Neural Machine Translation

## Setup

- Clone this repository

	`git clone https://github.com/redperiabras/filipineu.git`

- Setup virtual environment
	
	``
	pip install virtualenv
	virtualenv -p python3 venv
	venv/bin/activate (venv\scripts\activate on Windows)
	``

- Install the requirements and setup the development environment.

	`make install && make dev`

- Create the database.

	`python manage.py initdb`

- Run the application.

	`python terminal.py runserver`

- Navigate to `localhost:5000`.

## Terminal Commands

- Create Source Language Encoder
	
	``
	python terminal.py create-encoder
	--files <file addresses>
	--tokenizer <word/char>
	--hybrid
	--save-to <encoder filename>
	``

- Create Target Language Encoder

	``
	python terminal.py create-encoder 
	--files <file addresses> 
	--tokenizer <word/char> 
	--save-to <encoder filename>
	``

- Create Model

	``
	python terminal.py create-model
	--source-encoder <source encoder file>
	--target-encoder <target encoder file>
	--save-model <output model file>
	``

- Training Model

	``
	python terminal.py train
	--load-model <language model file>
	--train-data <training data file>
	--source-test-data <source test data file>
	--target-test-data <target test data file>
	``

## License

The MIT License (MIT). Please see the [license file](LICENSE) for more information.
