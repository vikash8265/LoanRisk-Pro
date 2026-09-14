const form =
    document.getElementById("loanForm");

const button =
    document.getElementById("predictButton");

const result =
    document.getElementById("result");

const placeholder =
    document.getElementById("placeholder");


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        button.disabled = true;

        button.textContent =
            "Analyzing...";


        const payload = {

            Age:
                Number(
                    document.getElementById("Age").value
                ),

            Income:
                Number(
                    document.getElementById("Income").value
                ),

            LoanAmount:
                Number(
                    document.getElementById("LoanAmount").value
                ),

            CreditScore:
                Number(
                    document.getElementById("CreditScore").value
                ),

            MonthsEmployed:
                Number(
                    document.getElementById("MonthsEmployed").value
                ),

            NumCreditLines:
                Number(
                    document.getElementById("NumCreditLines").value
                ),

            InterestRate:
                Number(
                    document.getElementById("InterestRate").value
                ),

            LoanTerm:
                Number(
                    document.getElementById("LoanTerm").value
                ),

            DTIRatio:
                Number(
                    document.getElementById("DTIRatio").value
                ),

            Education:
                document.getElementById("Education").value,

            EmploymentType:
                document.getElementById("EmploymentType").value,

            MaritalStatus:
                document.getElementById("MaritalStatus").value,

            HasMortgage:
                document.getElementById("HasMortgage").value,

            HasDependents:
                document.getElementById("HasDependents").value,

            LoanPurpose:
                document.getElementById("LoanPurpose").value,

            HasCoSigner:
                document.getElementById("HasCoSigner").value
        };


        try {

            const response =
                await fetch(
                    "/api/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                payload
                            )
                    }
                );


            if (!response.ok) {

                throw new Error(
                    await response.text()
                );
            }


            const data =
                await response.json();


            placeholder.classList.add(
                "hidden"
            );

            result.classList.remove(
                "hidden"
            );


            document.getElementById(
                "riskPercent"
            ).textContent =
                data.risk_percentage +
                "%";


            document.getElementById(
                "progressFill"
            ).style.width =
                Math.min(
                    data.risk_percentage,
                    100
                ) +
                "%";


            document.getElementById(
                "predictionText"
            ).textContent =
                data.prediction === 1
                    ? "Elevated repayment-failure risk"
                    : "Below model threshold";


            document.getElementById(
                "riskLevel"
            ).textContent =
                data.risk_level;


            document.getElementById(
                "threshold"
            ).textContent =
                data.threshold;


            document.getElementById(
                "message"
            ).textContent =
                data.message;


            document.getElementById(
                "disclaimer"
            ).textContent =
                data.disclaimer;


            const badge =
                document.getElementById(
                    "riskBadge"
                );


            badge.textContent =
                data.risk_level +
                " RISK";


            badge.className =
                "badge";


            if (data.risk_level === "Low") {

                badge.classList.add(
                    "risk-low"
                );
            }

            else if (
                data.risk_level === "Medium"
            ) {

                badge.classList.add(
                    "risk-medium"
                );
            }

            else {

                badge.classList.add(
                    "risk-high"
                );
            }

        }

        catch(error) {

            console.error(error);

            alert(
                "Prediction failed:\n" +
                error.message
            );
        }

        finally {

            button.disabled = false;

            button.textContent =
                "Analyze Loan Risk";
        }
    }
);
