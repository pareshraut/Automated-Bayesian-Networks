// A express servers which will handle api requests coming in and respond back with a json object , it will use a body parser as well as cross
const OpenAI = require('openai');
const { Configuration, OpenAIApi } = OpenAI

const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const port = 3001;

const configuration = new Configuration({
    organization: "org-1ginjSeQTcimiWLiIVe51wHW",
    apiKey: "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",
});
const openai = new OpenAIApi(configuration);

app.use(bodyParser.json());
app.use(cors());

app.post('/', async (req, res) => {
    const { message } = req.body;
    const response = await openai.createCompletion({
        model: "text-davinci-003",
        prompt: ` Pretend you are a Bayesian risk management chatbot and you are talking to a user.
        User : enters prompt.
        Bayesian risk management chatbot : parse the prompt entered by user and come up with a list of prospective nodes for bayesian risk model and give it back to the user.
        User: ${message}
        Bayesian risk management chatbot :`,
        max_tokens: 100,
        temperature: 0,
      });
      console.log(response.data)
        if(response.data.choices[0].text){
            res.json({ message : response.data.choices[0].text})
    }
});


app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`);
    }
);

