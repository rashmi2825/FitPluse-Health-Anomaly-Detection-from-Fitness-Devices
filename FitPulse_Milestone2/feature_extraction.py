{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "771355ca",
        "outputId": "d5b347a5-f7f2-4bb7-af4e-573057cb19c3"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from tsfresh import extract_features\n",
        "from tsfresh.feature_extraction import MinimalFCParameters\n",
        "from sklearn.feature_selection import VarianceThreshold\n"
      ],
      "metadata": {
        "id": "AwdLCmC01HgK"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/content/drive/MyDrive/FitPulse_Milestone1/data/minute_level_data.csv\")\n",
        "df[\"timestamp\"] = pd.to_datetime(df[\"timestamp\"])"
      ],
      "metadata": {
        "id": "0LWfF_Xy1i32"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.columns\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Xxy-XEoZB0pV",
        "outputId": "c70612ff-5ee2-47de-9f27-5753433c3f47"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['Id', 'timestamp', 'HeartRate', 'Date', 'TotalSteps', 'SleepFlag'], dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 38
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "user_ids = df[\"Id\"].unique()[:5]\n",
        "df = df[df[\"Id\"].isin(user_ids)]\n"
      ],
      "metadata": {
        "id": "_0e3gBXR1wT8"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = df.groupby(\"Id\").head(600)"
      ],
      "metadata": {
        "id": "G7oswuTH10D4"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ts_data = df[[\"Id\", \"timestamp\", \"HeartRate\"]].dropna()\n",
        "ts_data.columns = [\"id\", \"time\", \"value\"]"
      ],
      "metadata": {
        "id": "SCP9KKOj12x5"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "features = extract_features(\n",
        "    ts_data,\n",
        "    column_id=\"id\",\n",
        "    column_sort=\"time\",\n",
        "    default_fc_parameters=MinimalFCParameters(),\n",
        "    disable_progressbar=False\n",
        ")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EkpfQ_J414no",
        "outputId": "3f70bdc1-7317-44d8-87e9-927b480f3eb0"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tsfresh.feature_extraction.settings:Dependency not available for matrix_profile, this feature will be disabled!\n",
            "Feature Extraction: 100%|██████████| 5/5 [00:00<00:00, 920.65it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "features"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 226
        },
        "id": "qmOROzuECKQq",
        "outputId": "6b6f6bd8-91ca-4ef5-e92c-046b4e390492"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "            value__sum_values  value__median  value__mean  value__length  \\\n",
              "2022484408            50818.0           78.5    84.696667          600.0   \n",
              "2026352035            28877.0           63.0    65.779043          439.0   \n",
              "2347167796            40270.0           67.0    67.116667          600.0   \n",
              "4020332650            41919.0           70.0    69.865000          600.0   \n",
              "4558609924            48976.0           82.0    81.626667          600.0   \n",
              "\n",
              "            value__standard_deviation  value__variance  \\\n",
              "2022484408                  24.436816       597.157989   \n",
              "2026352035                   5.530937        30.591269   \n",
              "2347167796                   4.448939        19.793056   \n",
              "4020332650                   2.307258         5.323442   \n",
              "4558609924                   7.051522        49.723956   \n",
              "\n",
              "            value__root_mean_square  value__maximum  value__absolute_maximum  \\\n",
              "2022484408                88.151479           155.0                    155.0   \n",
              "2026352035                66.011164            80.0                     80.0   \n",
              "2347167796                67.263958            89.0                     89.0   \n",
              "4020332650                69.903088            75.0                     75.0   \n",
              "4558609924                81.930682           102.0                    102.0   \n",
              "\n",
              "            value__minimum  \n",
              "2022484408            53.0  \n",
              "2026352035            57.0  \n",
              "2347167796            59.0  \n",
              "4020332650            63.0  \n",
              "4558609924            61.0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-be3e752c-3fa2-4aaf-b628-14b4c86a9789\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>value__sum_values</th>\n",
              "      <th>value__median</th>\n",
              "      <th>value__mean</th>\n",
              "      <th>value__length</th>\n",
              "      <th>value__standard_deviation</th>\n",
              "      <th>value__variance</th>\n",
              "      <th>value__root_mean_square</th>\n",
              "      <th>value__maximum</th>\n",
              "      <th>value__absolute_maximum</th>\n",
              "      <th>value__minimum</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2022484408</th>\n",
              "      <td>50818.0</td>\n",
              "      <td>78.5</td>\n",
              "      <td>84.696667</td>\n",
              "      <td>600.0</td>\n",
              "      <td>24.436816</td>\n",
              "      <td>597.157989</td>\n",
              "      <td>88.151479</td>\n",
              "      <td>155.0</td>\n",
              "      <td>155.0</td>\n",
              "      <td>53.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2026352035</th>\n",
              "      <td>28877.0</td>\n",
              "      <td>63.0</td>\n",
              "      <td>65.779043</td>\n",
              "      <td>439.0</td>\n",
              "      <td>5.530937</td>\n",
              "      <td>30.591269</td>\n",
              "      <td>66.011164</td>\n",
              "      <td>80.0</td>\n",
              "      <td>80.0</td>\n",
              "      <td>57.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2347167796</th>\n",
              "      <td>40270.0</td>\n",
              "      <td>67.0</td>\n",
              "      <td>67.116667</td>\n",
              "      <td>600.0</td>\n",
              "      <td>4.448939</td>\n",
              "      <td>19.793056</td>\n",
              "      <td>67.263958</td>\n",
              "      <td>89.0</td>\n",
              "      <td>89.0</td>\n",
              "      <td>59.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4020332650</th>\n",
              "      <td>41919.0</td>\n",
              "      <td>70.0</td>\n",
              "      <td>69.865000</td>\n",
              "      <td>600.0</td>\n",
              "      <td>2.307258</td>\n",
              "      <td>5.323442</td>\n",
              "      <td>69.903088</td>\n",
              "      <td>75.0</td>\n",
              "      <td>75.0</td>\n",
              "      <td>63.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4558609924</th>\n",
              "      <td>48976.0</td>\n",
              "      <td>82.0</td>\n",
              "      <td>81.626667</td>\n",
              "      <td>600.0</td>\n",
              "      <td>7.051522</td>\n",
              "      <td>49.723956</td>\n",
              "      <td>81.930682</td>\n",
              "      <td>102.0</td>\n",
              "      <td>102.0</td>\n",
              "      <td>61.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-be3e752c-3fa2-4aaf-b628-14b4c86a9789')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-be3e752c-3fa2-4aaf-b628-14b4c86a9789 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-be3e752c-3fa2-4aaf-b628-14b4c86a9789');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-b092de84-7655-4e14-a4db-79d1f0ce5692\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-b092de84-7655-4e14-a4db-79d1f0ce5692')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-b092de84-7655-4e14-a4db-79d1f0ce5692 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "  <div id=\"id_00accba4-aedf-4bf2-8237-63a65365586d\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('features')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_00accba4-aedf-4bf2-8237-63a65365586d button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('features');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "features",
              "summary": "{\n  \"name\": \"features\",\n  \"rows\": 5,\n  \"fields\": [\n    {\n      \"column\": \"value__sum_values\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8681.681432764048,\n        \"min\": 28877.0,\n        \"max\": 50818.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          28877.0,\n          48976.0,\n          40270.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__median\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 7.940403012442126,\n        \"min\": 63.0,\n        \"max\": 82.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          63.0,\n          82.0,\n          67.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__mean\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8.724671326855258,\n        \"min\": 65.77904328018224,\n        \"max\": 84.69666666666667,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          65.77904328018224,\n          81.62666666666667,\n          67.11666666666666\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__length\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 72.00138887549323,\n        \"min\": 439.0,\n        \"max\": 600.0,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          439.0,\n          600.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__standard_deviation\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8.93492185269133,\n        \"min\": 2.3072584741781026,\n        \"max\": 24.436816259261125,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          5.530937465216948,\n          7.051521506423671\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__variance\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 255.78291816877555,\n        \"min\": 5.323441666666666,\n        \"max\": 597.1579888888889,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          30.591269244140484,\n          49.723955555555555\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__root_mean_square\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.83634255505673,\n        \"min\": 66.01116423833341,\n        \"max\": 88.15147947330965,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          66.01116423833341,\n          81.93068208349462\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__maximum\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 32.30634612580011,\n        \"min\": 75.0,\n        \"max\": 155.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          80.0,\n          102.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__absolute_maximum\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 32.30634612580011,\n        \"min\": 75.0,\n        \"max\": 155.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          80.0,\n          102.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"value__minimum\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3.847076812334269,\n        \"min\": 53.0,\n        \"max\": 63.0,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          57.0,\n          61.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "# Create the 'data' directory if it doesn't exist\n",
        "os.makedirs('data', exist_ok=True)\n",
        "\n",
        "features.to_csv(\"data/tsfresh_features.csv\")\n",
        "selected_features.to_csv(\"data/selected_features.csv\")\n",
        "\n",
        "print(\"Final selected feature shape:\", selected_features.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "90EH7NaA2O3t",
        "outputId": "fe7e7e4f-5b94-4ac5-e6b8-a46fbb9a9fe2"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Final selected feature shape: (5, 10)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Kkz1LJAg2RGU"
      },
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fb7561ee"
      },
      "source": [
        "# Task\n",
        "Perform feature extraction on the minute-level heart rate data from \"/content/drive/MyDrive/FitPulse_Milestone1/data/minute_level_data.csv\" for 15 unique users, grouping each user's data by the first 600 entries, extracting features using `tsfresh` with `MinimalFCParameters`, applying variance thresholding for feature selection, and saving the extracted and selected features to \"data/tsfresh_features.csv\" and \"data/selected_features.csv\" respectively."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dcc7d3ac"
      },
      "source": [
        "## Filter DataFrame for 15 Users\n",
        "\n",
        "### Subtask:\n",
        "Re-run the cell to filter the main DataFrame `df` for the first 15 unique user IDs.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dbc4dc9e"
      },
      "source": [
        "**Reasoning**:\n",
        "The subtask explicitly asks to re-run the code cell with ID `_0e3gBXR1wT8` to filter the `df` DataFrame for the first 15 unique user IDs. This command will execute that cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "871f1529"
      },
      "source": [
        "user_ids = df[\"Id\"].unique()[:15]\n",
        "df = df[df[\"Id\"].isin(user_ids)]"
      ],
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e38db336"
      },
      "source": [
        "**Reasoning**:\n",
        "The subtask explicitly asks to re-run the code cell with ID `_0e3gBXR1wT8` to filter the `df` DataFrame for the first 15 unique user IDs. This command will execute that cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6ef6d9e9"
      },
      "source": [
        "user_ids = df[\"Id\"].unique()[:15]\n",
        "df = df[df[\"Id\"].isin(user_ids)]"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5d190878"
      },
      "source": [
        "**Reasoning**:\n",
        "The subtask explicitly asks to re-run the code cell with ID `_0e3gBXR1wT8` to filter the `df` DataFrame for the first 15 unique user IDs. This command will execute that cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "072889e9"
      },
      "source": [
        "user_ids = df[\"Id\"].unique()[:15]\n",
        "df = df[df[\"Id\"].isin(user_ids)]"
      ],
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "da15cc93"
      },
      "source": [
        "**Reasoning**:\n",
        "The subtask explicitly asks to re-run the code cell with ID `_0e3gBXR1wT8` to filter the `df` DataFrame for the first 15 unique user IDs. This command will execute that cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3348503a"
      },
      "source": [
        "user_ids = df[\"Id\"].unique()[:15]\n",
        "df = df[df[\"Id\"].isin(user_ids)]"
      ],
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aaa57470"
      },
      "source": [
        "# Task\n",
        "Perform feature extraction on the minute-level heart rate data from \"/content/drive/MyDrive/FitPulse_Milestone1/data/minute_level_data.csv\" for 15 unique users, grouping each user's data by the first 600 entries, extracting features using `tsfresh` with `MinimalFCParameters`, applying variance thresholding for feature selection, and saving the extracted and selected features to \"data/tsfresh_features.csv\" and \"data/selected_features.csv\" respectively."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ab6b7813"
      },
      "source": [
        "## Filter DataFrame for 15 Users\n",
        "\n",
        "### Subtask:\n",
        "Re-running the cell to filter the main DataFrame `df` for the first 15 unique user IDs. This ensures the subsequent steps operate on the larger dataset.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "14f879b8"
      },
      "source": [
        "**Reasoning**:\n",
        "The subtask explicitly asks to re-run the code cell with ID `_0e3gBXR1wT8` to filter the `df` DataFrame for the first 15 unique user IDs. This command will execute that cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0ddb44e3"
      },
      "source": [
        "user_ids = df[\"Id\"].unique()[:15]\n",
        "df = df[df[\"Id\"].isin(user_ids)]"
      ],
      "execution_count": 37,
      "outputs": []
    }
  ]
}