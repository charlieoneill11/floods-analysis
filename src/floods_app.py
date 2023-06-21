from flask import Flask, render_template, request, render_template_string, url_for, jsonify
from flask import escape
import matplotlib
matplotlib.use('Agg')
import time

from main import *
from strategy import Strategy

app = Flask(__name__)

def round_to_millions(value):
    if abs(value) > 10000:  # Assuming if it's >10000, it's not rounded
        return round(round(value / 1e6, 1), 1)
    else:
        return round(value, 1)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/parameter_estimation')
def parameter_estimation():
    return render_template('parameter_estimation.html')

@app.route('/strategy_tables')
def strategy_tables():
    # Generate the dataframes
    years = range(2022, 2043)
    strategies = ["../data/compulsory_buyback.txt", "../data/voluntary_buyback.txt", "../data/voluntary_landswap.txt"]
    strategy = Strategy(years)
    strategy_dfs = strategy.calculate_strategies(strategies)

    # Convert each dataframe to HTML and store in a dict with the strategy as the key
    tables = {}
    for strategy, df in zip(strategies, strategy_dfs):
        title = strategy.split('/')[-1].split('.')[0].replace('_', ' ').title()
        tables[title] = df.to_html(classes="strategy-table", escape=False, justify="center", border=0)
    
    # Render the tables in the template
    template = '''
        <h2>Strategy Alternatives</h2>
        {% for title, table in tables.items() %}
            <h3>{{ title }}</h3>
            {{ table|safe }}
        {% endfor %}
    '''
    return render_template_string(template, tables=tables)

@app.route('/cost_estimation', methods=['GET', 'POST'])
def cost_estimation():
    if request.method == 'POST':
        # Retrieve user input from the form
        hh_in_1_to_10 = int(request.form.get('hh_in_1_to_10', 19))
        hh_in_1_to_100 = int(request.form.get('hh_in_1_to_100', 855))
        hh_in_1_to_1000 = int(request.form.get('hh_in_1_to_1000', 1959))

        # Generate the plot
        generate_plot(hh_in_1_to_10, hh_in_1_to_100, hh_in_1_to_1000)

        # Create an instance of the FloodsAnalysis class with user input (or default values)
        flood_analysis = FloodsAnalysis(hh_in_1_to_10=hh_in_1_to_10, 
                                        hh_in_1_to_100=hh_in_1_to_100, 
                                        hh_in_1_to_1000=hh_in_1_to_1000)

        # Calculate the results
        repair_costs = flood_analysis.calculate_repair_costs()
        rebuild_costs = flood_analysis.calculate_rebuild_costs()
        income_costs = flood_analysis.calculate_income_costs()
        rental_costs = flood_analysis.calculate_rental_costs()
        total_costs = flood_analysis.calculate_total_costs()

        results = {
            'Repair': repair_costs,
            'Rebuild': rebuild_costs,
            'Income': income_costs,
            'Rental': rental_costs,
            'Total': total_costs,
        }

        timestamp = int(time.time())
        plot_url = url_for('static', filename='plot.png', t=timestamp)

        # Create an instance of the NPV class called npv_instance
        npv_instance = NPV(hh_in_1_to_10=hh_in_1_to_10, 
                           hh_in_1_to_100=hh_in_1_to_100, 
                           hh_in_1_to_1000=hh_in_1_to_1000)
        npv_table_df = npv_instance.npv_table()
        npv_table_html = npv_table_df.to_html(classes="npv-table", escape=False, justify="center", border=0)

        # Generate the dataframes
        years = range(2022, 2043)
        strategies = ["../data/compulsory_buyback.txt", "../data/voluntary_buyback.txt", "../data/voluntary_landswap.txt"]
        strategy_obj = Strategy(years)
        strategy_dfs = strategy_obj.calculate_strategies(strategies)

        # Convert each summary dataframe to HTML and store in a dict with the strategy as the key
        tables = {}
        for strategy, df in zip(strategies, strategy_dfs):
            title = strategy.split('/')[-1].split('.')[0].replace('_', ' ').title()
            summary_df = strategy_obj.summarise_strategy(df)
            summary_df = summary_df.applymap(round_to_millions)
            tables[title] = summary_df.to_html(classes="strategy-table", escape=False, justify="center", border=0)

        
        table_template = '''
            <h2 style="text-align:center;">Net Present Value (NPV) Costs</h2>
            {{ npv_table_html|safe }}

            <h2 style="text-align:center;">Flood Cost Estimation</h2>
            <table style="margin-left: auto; margin-right: auto; border-collapse: separate; border-spacing: 15px;">
                <thead>
                    <tr>
                        <th></th>
                        <th>1-in-10</th>
                        <th>1-in-100</th>
                        <th>1-in-1000</th>
                        <th>Annual exp. cost</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row, values in results.items() %}
                        <tr>
                            <td>{{ row }}</td>
                            {% for value in values %}
                                <td>{{ (value / 1000000)|round(2) }} M$</td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
            <div style="display: flex; justify-content: center;">
                <img src="{{ plot_url }}" alt="Plot">
            </div>
        '''

        strategy_template = '''
            <h2>Strategy Alternatives</h2>
            {% for title, table in tables.items() %}
                <h3>{{ title }}</h3>
                {{ table|safe }}
            {% endfor %}
        '''

        return jsonify({
            'html': render_template_string(table_template, results=results, plot_url=plot_url, npv_table_html=npv_table_html) +
                    render_template_string(strategy_template, tables=tables)
        })


    # Generate the dataframes
    years = range(2022, 2043)
    strategies = ["../data/compulsory_buyback.txt", "../data/voluntary_buyback.txt", "../data/voluntary_landswap.txt"]
    strategy_obj = Strategy(years)
    strategy_dfs = strategy_obj.calculate_strategies(strategies)

    # Convert each summary dataframe to HTML and store in a dict with the strategy as the key
    tables = {}
    for strategy, df in zip(strategies, strategy_dfs):
        title = strategy.split('/')[-1].split('.')[0].replace('_', ' ').title()
        summary_df = strategy_obj.summarise_strategy(df)
        summary_df = summary_df.applymap(round_to_millions)
        tables[title] = summary_df.to_html(classes="strategy-table", escape=False, justify="center", border=0)
    
    return render_template('cost_estimation.html', tables=tables)

@app.route('/calculate_costs', methods=['POST'])
def calculate_costs():
    # Your existing code for the 'result' route, update the route name

    # Retrieve user input from the form
    hh_in_1_to_10 = int(request.form.get('hh_in_1_to_10', 19))
    hh_in_1_to_100 = int(request.form.get('hh_in_1_to_100', 855))
    hh_in_1_to_1000 = int(request.form.get('hh_in_1_to_1000', 1959))

    # Generate the plot
    generate_plot(hh_in_1_to_10, hh_in_1_to_100, hh_in_1_to_1000)

    # Create an instance of the FloodsAnalysis class with user input (or default values)
    flood_analysis = FloodsAnalysis(hh_in_1_to_10=hh_in_1_to_10, 
                                    hh_in_1_to_100=hh_in_1_to_100, 
                                    hh_in_1_to_1000=hh_in_1_to_1000)

    # Calculate the results
    repair_costs = flood_analysis.calculate_repair_costs()
    rebuild_costs = flood_analysis.calculate_rebuild_costs()
    income_costs = flood_analysis.calculate_income_costs()
    rental_costs = flood_analysis.calculate_rental_costs()
    total_costs = flood_analysis.calculate_total_costs()

    results = {
        'Repair': repair_costs,
        'Rebuild': rebuild_costs,
        'Income': income_costs,
        'Rental': rental_costs,
        'Total': total_costs,
    }

    # Add timestamp to the image URL
    timestamp = int(time.time())
    plot_url = url_for('static', filename='plot.png', t=timestamp)

    table_template = '''
    <table style="margin-left: auto; margin-right: auto; border-collapse: separate; border-spacing: 15px;">
        <thead>
            <tr>
                <th></th>
                <th>1-in-10</th>
                <th>1-in-100</th>
                <th>1-in-1000</th>
                <th>Annual exp. cost</th>
            </tr>
        </thead>
        <tbody>
            {% for row, values in results.items() %}
                <tr>
                    <td>{{ row }}</td>
                    {% for value in values %}
                        <td>{{ (value / 1000000)|round(2) }} M$</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </tbody>
    </table>
    <img src="{{ plot_url }}" alt="Plot">
    '''

    return render_template_string(table_template, results=results, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True, port=5001)