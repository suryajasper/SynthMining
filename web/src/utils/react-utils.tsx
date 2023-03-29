import React from "react";
import { useNavigate } from "react-router-dom";

export function withNavigate(Component) {
  return props => <Component {...props} navHook={useNavigate()} />;
}