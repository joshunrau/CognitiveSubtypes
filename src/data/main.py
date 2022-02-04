from dataset import Dataset


def main():
    data = Dataset()
    data.recode_diagnoses()
    data.apply_inclusion_criteria(method="OR")
    data.apply_exclusion_criteria()
    data.recode_vars()
    data.write_csv()


if __name__ == "__main__":
    main()