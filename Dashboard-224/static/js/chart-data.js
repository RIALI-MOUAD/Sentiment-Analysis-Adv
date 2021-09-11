var randomScalingFactor = function(){ return Math.round(Math.random()*1000)};
//['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
	var lineChartData = {
		labels : [0,1],
		datasets : [
			{
				label: "Angry",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(255,0,0,1)",
				pointColor : "rgba(255,0,0,1)",
				/**
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(255,0,0,1)",
				**/
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Disgust",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(255, 127, 0, 1)",
				pointColor : "rgba(255, 127, 0, 1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(255, 127, 0, 1)",
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Fear",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(255,255,0,1)",
				pointColor : "rgba(255,255,0,1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(255,255,0,1)",
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Happy",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(0,255,0,1)",
				pointColor : "rgba(0,255,0,1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(0,255,220,1)",
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Neutral",
				fill:false,
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(153, 235, 255,1)",

				pointColor : "rgba(153, 235, 255,1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(153, 235, 255,1)",
				/****/
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Sad",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(139,0,255,1)",
				pointColor : "rgba(139,0,255,1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(139,0,255,1)",
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			},
			{
				label: "Surprise",
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(46,43,95,1)",
				pointColor : "rgba(46,43,95,1)",
				pointStrokeColor : "#fff",
				pointHighlightFill : "#fff",
				pointHighlightStroke : "rgba(46,43,95,1)",
				//data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
        data :[]
			}
		]

	}

	var barChartData = {
		labels : ["January","February","March","April","May","June","July"],
		datasets : [
			{
				fillColor : "rgba(255,255,255,0.1)",
				strokeColor : "rgba(220,220,220,0.8)",
				highlightFill: "rgba(220,220,220,0.75)",
				highlightStroke: "rgba(220,220,220,1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			}/**,
			{
				fillColor : "rgba(48, 164, 255, 0.2)",
				strokeColor : "rgba(48, 164, 255, 0.8)",
				highlightFill : "rgba(48, 164, 255, 0.75)",
				highlightStroke : "rgba(48, 164, 255, 1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			}**/
		]

	}

	var pieData = [
			{
				value: 300,
				color:"#30a5ff",
				highlight: "#62b9fb",
				label: "Blue"
			},
			{
				value: 50,
				color: "#ffb53e",
				highlight: "#fac878",
				label: "Orange"
			},
			{
				value: 100,
				color: "#1ebfae",
				highlight: "#3cdfce",
				label: "Teal"
			},
			{
				value: 120,
				color: "#f9243f",
				highlight: "#f6495f",
				label: "Red"
			}

		];

	var doughnutData = [
				{
					value: 300,
					color:"#30a5ff",
					highlight: "#62b9fb",
					label: "Blue"
				},
				{
					value: 50,
					color: "#ffb53e",
					highlight: "#fac878",
					label: "Orange"
				},
				{
					value: 100,
					color: "#1ebfae",
					highlight: "#3cdfce",
					label: "Teal"
				},
				{
					value: 120,
					color: "#f9243f",
					highlight: "#f6495f",
					label: "Red"
				}

			];

	var radarData = {
	    labels: ["Eating", "Drinking", "Sleeping", "Designing", "Coding", "Cycling", "Running"],
	    datasets: [
	        {
	            label: "My First dataset",
	            fillColor: "rgba(220,220,220,0.2)",
	            strokeColor: "rgba(220,220,220,1)",
	            pointColor: "rgba(220,220,220,1)",
	            pointStrokeColor: "#fff",
	            pointHighlightFill: "#fff",
	            pointHighlightStroke: "rgba(220,220,220,1)",
	            data: [65, 59, 90, 81, 56, 55, 40]
	        },
	        {
	            label: "My Second dataset",
	            fillColor : "rgba(48, 164, 255, 0.2)",
	            strokeColor : "rgba(48, 164, 255, 0.8)",
	            pointColor : "rgba(48, 164, 255, 1)",
	            pointStrokeColor : "#fff",
	            pointHighlightFill : "#fff",
	            pointHighlightStroke : "rgba(48, 164, 255, 1)",
	            data: [28, 48, 40, 19, 96, 27, 100]
	        }
	    ]
	};

	var polarData = [
		    {
		    	value: 300,
		    	color: "#1ebfae",
		    	highlight: "#38cabe",
		    	label: "Teal"
		    },
		    {
		    	value: 140,
		    	color: "#ffb53e",
		    	highlight: "#fac878",
		    	label: "Orange"
		    },
		    {
		    	value: 220,
		    	color:"#30a5ff",
		    	highlight: "#62b9fb",
		    	label: "Blue"
		    },
		    {
		    	value: 250,
		    	color: "#f9243f",
		    	highlight: "#f6495f",
		    	label: "Red"
		    }

	];
	function addData(chart, label, data) {
	    chart.data.labels.push(label);
	    chart.data.datasets.forEach((dataset) => {
	        dataset.data.push(data);
	    });
	    chart.update();
	}
