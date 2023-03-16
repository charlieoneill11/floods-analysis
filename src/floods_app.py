from flask import Flask, render_template, request, render_template_string, url_for, jsonify
import matplotlib
matplotlib.use('Agg')
import time

from main import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/parameter_estimation')
def parameter_estimation():
    return render_template('parameter_estimation.html')

@app.route('/cost_estimation', methods=['GET', 'POST'])
def cost_estimation():
    # Your existing code for the 'result' route, update the route name
    # ...
    
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

        return jsonify({
            'html': render_template_string(table_template, results=results, plot_url=plot_url)
        })

    return render_template('cost_estimation.html')

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
    app.run(debug=True)