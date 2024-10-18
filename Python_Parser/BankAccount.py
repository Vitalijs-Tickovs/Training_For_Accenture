class BankAccount:
    def __init__(self, owner, balance=0):
        self.account = owner                     # Name of the account holder
        self.account_balance = balance           # Balance
        self.account_details = {}                # Dictionary for account details

    def set_account_details(self):
        """Set the details of the account in a dictionary."""
        first_name, last_name = self.account.split(" ")
        self.account_details['first_name'] = first_name
        self.account_details['last_name'] = last_name
        self.account_details['initials'] = first_name[0] + last_name[0]
        self.account_details['full_name'] = self.account
        self.account_details['account_balance'] = self.account_balance

    def display_account_details(self):
        """Display the account details."""
        print(f"Account Name: {self.account}")
        print(f"Account Balance: {self.account_balance}")
        for key, value in self.account_details.items():
            print(f"{key}: {value}")


# Create an instance of BankAccount
account1 = BankAccount("John Doe", 100)

# Set the account details
account1.set_account_details()

# Display the account details
account1.display_account_details()
